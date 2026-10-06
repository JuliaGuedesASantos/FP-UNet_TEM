import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

####################################################################################################
########################################  CONJUNTOS  FUZZY  ########################################
####################################################################################################

class ConjsFuzzy:
    """ Calcula a pertinência de cada conjunto fuzzy (P, M, G) para um dado tensor de entrada x."""
    
    def __init__(self, rmax=6):
        """ Inicializa a classe de Conjuntos Fuzzy.

        Args:
            rmax (int): valor máximo do universo E, considerando E = [0, rmax] baseado nas saídas de uma Clipped ReLU (default: 6)
        """

        if (
            isinstance(rmax, bool)
            or not isinstance(rmax, numbers.Real)
            or not math.isfinite(rmax)
            or rmax <= 0
        ):
            raise ValueError("rmax deve ser um número finito maior que zero.")

        self.rmax = rmax # Teto da função de ativação ReLU6, cuja saída é o universo considerado.


    def P(self,x):
        """ Calcula a pertinência do conjunto fuzzy P (pequeno) para um dado tensor de entrada x.

        Args: 
            x (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de valores de entrada.
            
        Returns:
            pert_P (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de pertinência do conjunto fuzzy P para cada elemento do tensor de entrada x.
        
        
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
        
        """

        d = self.rmax / 2
        c = d / 3

        pert_P = torch.where( # Se
            x < c, # x é baixo, região de pertinência máxima do conjunto P,
            torch.ones_like(x), # preenche com pertinência 1;
            torch.where( # Se não,
                x <= d, # x é médio-baixo, pertinência linear decrescente para P,
                (d - x) / (d - c), # calcula a pertinência;
                torch.zeros_like(x) # Caso contrário, preenche com pertinência 0.
                )
            )

        return pert_P


    def M(self,x):
        """ Calcula a pertinência do conjunto fuzzy M (médio) para um dado tensor de entrada x.

        Args: 
            x (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de valores de entrada.
            
        Returns:
            pert_M (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de pertinência do conjunto fuzzy M para cada elemento do tensor de entrada x.
       
            
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
            
        """

        a = self.rmax / 4
        m = self.rmax / 2
        b = m + a

        pert_M = torch.where( # Se
            (x >= a) & (x <= m), # x é médio-baixo, pertinência linear crescente para M,
            (x - a) / (m - a), # calcula a pertinência;
            torch.where( # Se não, 
                (x > m) & (x < b), # x é médio-alto, pertinência linear decrescente para M,
                (b - x) / (b - m), # calcula a pertinência;
                torch.zeros_like(x) # Caso contrário, preenche com pertinência 0.
            )
        )

        return pert_M


    def G(self,x):
        """ Calcula a pertinência do conjunto fuzzy G (grande) para um dado tensor de entrada x.

        Args: 
            x (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de valores de entrada.
            
        Returns:
            pert_G (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de pertinência do conjunto fuzzy G para cada elemento do tensor de entrada x.
        
            
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
        
        """

        r = self.rmax / 2
        q = r + self.rmax / 4

        pert_G = torch.where( # Se
            x < r, # x é pequeno, região de pertinência mínima do conjunto G,
            torch.zeros_like(x), # preenche com pertinência 0;
            torch.where( # Se não,
                x <= q, # x é médio-alto, pertinência linear crescente para G,
                (x - r) / (q - r), # calcula a pertinência;
                torch.ones_like(x) # Caso contrário, preecnhe com pertinência 1.
            )
        )

        return pert_G


    def PMG(self,x):
        """ Calcula a pertinência dos conjuntos fuzzy P, M e G para um dado tensor de entrada x.

        Args: 
            x (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de valores de entrada.
            
        Returns:
            self.pertinencias (torch.Tensor ~ (N, Z, V, kh*kw, H_out*W_out)): tensor de tensores de pertinências para cada elemento do tensor de entrada x em cada conjunto P, M e G.
        
        
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        V: número de conjuntos fuzzy (P, M, G);
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
        
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("x deve ser um torch.Tensor.")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x deve conter apenas valores finitos.")
        if bool(((x < 0) | (x > self.rmax)).any()):
            raise ValueError(f"x deve pertencer ao universo fuzzy [0, {self.rmax}].")

        # Concatena as pertiências de cada conjunto fuzzy em um único tensor
        pertinencias = torch.stack(
            [
                self.P(x),
                self.M(x),
                self.G(x)
            ],
            dim=2 # Aloca as pertinências por canal
        )

        return pertinencias


    ############################# Forward #############################

    def __call__(self, x):
        """ Permite chamar a classe como uma função, calculando as pertinências em P, M e G como padrão.

        Args: 
            x (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de valores de entrada.
            
        Returns:
            pertinencias (torch.Tensor  ~ (N, Z, V, kh*kw, H_out*W_out)): tensor de tensores de pertinências para cada elemento do tensor de entrada x em cada conjunto P, M e G.
        
        
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        V: número de conjuntos fuzzy (P, M, G); 
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
        
        """

        return self.PMG(x)


    def forward(self,x):
        """ Executa a classe de conjuntos fuzzy, calculando as pertinências em P, M e G como padrão.

        Args: 
            x (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de valores de entrada.
            
        Returns:
            pertinencias (torch.Tensor ~ (N, Z, V, kh*kw, H_out*W_out)): tensor de tensores de pertinências para cada elemento do tensor de entrada x em cada conjunto P, M e G.
        
        
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        V: número de conjuntos fuzzy (P, M, G);
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
        
        """

        return self.PMG(x)



####################################################################################################
#########################################  FUZZY  POOLING  #########################################
####################################################################################################

class FuzzyPooling(nn.Module):
    """ Reduz a dimensionalidade de um lote de imagens usando Fuzzy Pooling."""

    def __init__(self, tamanho_kernel=3, stride=2, funcs_pertinencia=None, padding="same"):
        """ Inicializa classe de Fuzzy Pooling.
        
        Args:        
            tamanho_kernel (int): Tamanho do kernel de pooling.
            stride (int): Tamanho do passo da janela de pooling.
            funcs_pertinencia (class or function ~ 1 parametro de entrada): Classe/função que calcula a função de pertinência de cada conjunto.                
            padding ("same" or int): Preenchimento das bordas. "same" preserva
                ceil(H/stride) x ceil(W/stride), como descrito no artigo.
        """

        super(FuzzyPooling, self).__init__()

        if (
            isinstance(tamanho_kernel, bool)
            or not isinstance(tamanho_kernel, numbers.Integral)
            or tamanho_kernel <= 0
        ):
            raise ValueError("tamanho_kernel deve ser um inteiro maior que zero.")
        if isinstance(stride, bool) or not isinstance(stride, numbers.Integral) or stride <= 0:
            raise ValueError("stride deve ser um inteiro maior que zero.")
        if padding != "same" and (
            isinstance(padding, bool)
            or not isinstance(padding, numbers.Integral)
            or padding < 0
        ):
            raise ValueError('padding deve ser "same" ou um inteiro não negativo.')

        if funcs_pertinencia is None:
            funcs_pertinencia = ConjsFuzzy()
        if not callable(funcs_pertinencia):
            raise TypeError("funcs_pertinencia deve ser chamável.")

        self.tamanho_kernel = int(tamanho_kernel) # Tamanho da janela de pooling
        self.stride = int(stride) # Passo da janela de pooling
        self.padding = padding
        self.funcs_pertinencia = funcs_pertinencia #conjuntos fuzzy definidos


    ########################## Soma Algébrica ##########################

    def somatorio_algebrico(self, matriz_pi):
        """ Calcula o somatório algébrico como agragação dos valores de patch fuzzy.

        Args:        
            matriz_pi (torch.Tensor ~ (N, Z, V, kh*kw, H_out*W_out)): tensor de pertinências para cada patch.

        Returns:
            soma (torch.Tensor ~ (N, Z, V, H_out*W_out)): tensor de somatório algébrico das pertinências para cada patch.         


        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        V: número de conjuntos fuzzy (P, M, G);
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.

        """

        # S = x + y - xy
        # S = 1 - (1 - x)(1 - y)
        # Somando o elementos de matriz_pi ==> S = 1 - prod(1 - matriz_pi)

        soma = 1 - torch.prod(1 - matriz_pi, dim=3) # Agrega os elementos do patch (dim=3 ~ kh*kw)

        return soma


    ########################## Fuzzificação ###########################

    def constroi_patch_fuzzy(self, patches, kh, kw): #por canal
        """ Calcula o correspondente fuzzy dos patches de entrada, selecionando o conjunto fuzzy de maior pertinência para cada patch.

        Args:        
            patches (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor contendo cada patch do dataset de imagens de entrada.
            kh (int): altura do kernel.
            kw (int): largura do kernel.

        Returns:
            patches_fuzzy (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor de pertinências no melhor conjunto fuzzy para cada patch.   


        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
                  
        """

        # Calcula a pertinência do patch para cada conjunto fuzzy (torch.Tensor ~ (N, Z, V, kh*kw, H_out*W_out)
        self.matrizes_pi = self.funcs_pertinencia(patches)

        # Calcula o agraga os valores de cada conjunto fuzzy para cada patch (torch.Tensor ~ (N, Z, V, H_out*W_out))
        self.scores = self.somatorio_algebrico(self.matrizes_pi)

        # Seleciona os índices do conjunto fuzzy de maior pertinência para cada patch (torch.Tensor ~ (N, Z, H_out*W_out))
        indices = torch.argmax(self.scores, dim=2)

        # Compatibilidade de dimensão com matrizes_pi
        indices = indices.unsqueeze(2) # Adiciona uma dimensão correspondente aos conjuntos fuzzy (apenas 1, o mais pertinente)
        indices = indices.unsqueeze(3) # Adiciona uma dimensão correspondente a elementos de patches

        self.indices = indices.expand(-1, -1, -1, kh * kw, -1) # Aumenta o número de elementos na dimensão de elementos de patches ~~~ mesmo conjunto fuzzy para todo o patch

        patches_fuzzy = torch.gather(self.matrizes_pi, dim=2, index=self.indices) # Resgata os valores do melhor patch fuzzy para cada patch (dim = 2)

        # Compatibilidade de dimensão com patches
        patches_fuzzy = patches_fuzzy.squeeze(2) # Remove a dimensão correspondente aos conjuntos fuzzy (apenas 1, o mais pertinente)

        return patches_fuzzy
    

    ######################### Defuzzificação ##########################

    def centro_gravidade(self, patches, patches_fuzzy):
        """ Calcula o centro de gravidade de cada patch fuzzy para defuzzificação.

        Args:        
            patches (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor contendo cada patch do dataset de imagens de entrada.
            patches_fuzzy (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out)): tensor contendo cada patch fuzzy do dataset de imagens de entrada.

        Returns:
            cog (torch.Tensor ~ (N, Z, H_out*W_out)): tensor de valores defuzzificados (centro de gravidade) para cada patch.    

        
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        kh*kw: número de elementos do patch (kernel);
        H_out*W_out: número de patches (janelas) na imagem de saída.
        
        """
        
        # CoG = S(pi*p)/S(pi)
        num = torch.sum(patches * patches_fuzzy, dim=2)
        den = torch.sum(patches_fuzzy, dim=2)

        cog = num / den.clamp_min(1e-12) # Evita divisão por zero

        return cog
    

    ############################# Preprocessamento #############################

    def _prepara_padding(self, H, W):
        """Retorna (esquerda, direita, topo, base), H_out e W_out."""

        if self.padding == "same":
            H_out = math.ceil(H / self.stride)
            W_out = math.ceil(W / self.stride)
            pad_h = max((H_out - 1) * self.stride + self.tamanho_kernel - H, 0)
            pad_w = max((W_out - 1) * self.stride + self.tamanho_kernel - W, 0)
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            return padding, H_out, W_out

        pad = int(self.padding)
        H_out = (H + 2 * pad - self.tamanho_kernel) // self.stride + 1
        W_out = (W + 2 * pad - self.tamanho_kernel) // self.stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                "A entrada espacial é menor que o kernel depois da aplicação do padding."
            )
        return (pad, pad, pad, pad), H_out, W_out


    ############################# Forward #############################

    def forward(self, x):
        """ Executa a classe de fuzzy pooling, reduzindo a dimensionalidade de imagens consiente de incerteza.

        Args: 
            x (array-like ~ (N, Z, H, W)): matriz de valores de entrada.
            
        Returns:
            valores_fuzzy (torch.Tensor ~ (N, Z, H_out, W_out)): tensor de imagens redimensionalizadas com base em conjuntos fuzzy.

            
        ** Dimensãos tensores:
        N: número de imagens/tamanho do lote;
        Z: número de Canais;
        H, W: altura e largura das imagens de entrada;
        H_out, W_out: altura e largura das umagens de saída.
        
        """

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.is_complex():
            raise TypeError("x deve conter valores reais.")
        if not x.is_floating_point():
            x = x.to(dtype=torch.get_default_dtype())
        if x.ndim != 4:
            raise ValueError("x deve ter quatro dimensões no formato (N, Z, H, W).")
        if any(dim <= 0 for dim in x.shape):
            raise ValueError("Todas as dimensões de x devem ser maiores que zero.")

        self.x = x


        N, Z, H, W = x.shape # Tamanho das dimensões

        kh, kw = self.tamanho_kernel, self.tamanho_kernel # Altura e largura do kernel

        # Cálculo do padding e do tamanho da imagem de saída
        padding, H_out, W_out = self._prepara_padding(H, W)
        x_padded = F.pad(self.x, padding) if any(padding) else self.x

        # Criação de patches/janelas de convolução (torch.Tensor ~ (N, Z*kh*kw, H_out*W_out))
        unfold = torch.nn.Unfold(kernel_size=(self.tamanho_kernel, self.tamanho_kernel), stride=self.stride)
        patches = unfold(x_padded)

        # Compatibilidade de dimensão (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out))
        self.patches = patches.view(N, Z, kh * kw, H_out * W_out)

        # Fuzzificação (torch.Tensor ~ (N, Z, kh*kw, H_out*W_out))
        self.patches_fuzzy = self.constroi_patch_fuzzy(self.patches, kh, kw)
       
       # Defuzzificação (torch.Tensor ~ (N, Z, H_out*W_out))
        valores_fuzzy = self.centro_gravidade(self.patches, self.patches_fuzzy)

        # Compatibilidade de dimensão (torch.Tensor ~ (N, Z, H_out, W_out))
        self.valores_fuzzy = valores_fuzzy.view(N, Z, H_out, W_out) 
        
        return self.valores_fuzzy
