import torch
from fuzzy_pooling import ConjsFuzzy, FuzzyPooling

import pytest

# To execute the tests, run the following command in the Anacorda terminal:
### pytest -q test_fuzzypooling.py

class TestFuzzyPooling:

    @pytest.fixture(autouse=True)
    def setup(self):
        # Sem padding para isolar exatamente o patch 3x3 usado no artigo.
        self.FzzPool = FuzzyPooling(padding=0)

        p1 = torch.tensor([[3/2, 2, 5/2],[3/2, 5/2, 7/2],[9/2, 4, 9/2]])
        self.tensor = p1.unsqueeze(0).unsqueeze(0)  

        self.pool = self.FzzPool(self.tensor)


    def test_forward(self):
        assert torch.allclose(self.pool, torch.tensor([[[[77/18]]]]))


    def test_conjs(self):
        conjs = self.FzzPool.matrizes_pi
        conjs_real= torch.Tensor([[[[[3/4],
                                    [1/2],
                                    [1/4],
                                    [3/4],
                                    [1/4],
                                    [0],
                                    [0],
                                    [0],
                                    [0]],

                                    [[0],
                                    [1/3],
                                    [2/3],
                                    [0],
                                    [2/3],
                                    [2/3],
                                    [0],
                                    [1/3],
                                    [0]],

                                    [[0],
                                    [0],
                                    [0],
                                    [0],
                                    [0],
                                    [1/3],
                                    [1],
                                    [2/3],
                                    [1]]]]])
        
        assert torch.allclose(conjs, conjs_real)
    

    def test_scores(self):
        scores = self.FzzPool.scores
        scores_real = torch.Tensor([[[[503/512], [239/243],[1]]]])
        assert torch.allclose(scores, scores_real)


    def test_patch_fuzzy(self):
        patch_fuzzy = self.FzzPool.patches_fuzzy
        patch_fuzzy_real = torch.Tensor([[[[0],
                                    [0],
                                    [0],
                                    [0],
                                    [0],
                                    [1/3],
                                    [1],
                                    [2/3],
                                    [1]]]])
        assert torch.allclose(patch_fuzzy, patch_fuzzy_real)


def test_same_padding_reduz_dimensoes_pela_metade():
    pool = FuzzyPooling()

    assert pool(torch.rand(2, 3, 28, 28)).shape == (2, 3, 14, 14)
    assert pool(torch.rand(2, 3, 27, 29)).shape == (2, 3, 14, 15)


@pytest.mark.parametrize("entrada", [
    torch.ones(1, 1, 3, 3, dtype=torch.long),
    [[[[1, 2, 3], [4, 5, 6], [1, 2, 3]]]],
])
def test_entrada_inteira_e_convertida_para_float(entrada):
    resultado = FuzzyPooling(padding=0)(entrada)

    assert resultado.is_floating_point()
    assert torch.isfinite(resultado).all()


def test_instancias_nao_compartilham_funcoes_de_pertinencia():
    primeiro = FuzzyPooling()
    segundo = FuzzyPooling()

    assert primeiro.funcs_pertinencia is not segundo.funcs_pertinencia


def test_limites_das_funcoes_de_pertinencia():
    valores = torch.tensor([0.0, 1.0, 1.5, 3.0, 4.5, 6.0]).view(1, 1, 1, -1)
    pertinencias = ConjsFuzzy()(valores)
    esperado = torch.tensor(
        [
            [1.0, 1.0, 0.75, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    ).view(1, 1, 3, 1, -1)

    assert torch.allclose(pertinencias, esperado)


def test_backward():
    entrada = (torch.rand(2, 3, 7, 7, dtype=torch.double) * 6).requires_grad_()
    resultado = FuzzyPooling()(entrada)

    resultado.sum().backward()

    assert entrada.grad is not None
    assert torch.isfinite(entrada.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tamanho_kernel": 0},
        {"stride": 0},
        {"padding": -1},
        {"padding": "invalid"},
        {"funcs_pertinencia": 42},
    ],
)
def test_rejeita_configuracao_invalida(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FuzzyPooling(**kwargs)


def test_rejeita_formato_de_entrada_invalido():
    with pytest.raises(ValueError, match="quatro dimensões"):
        FuzzyPooling()(torch.rand(3, 3))


@pytest.mark.parametrize("rmax", [0, -1, float("inf"), float("nan"), True])
def test_rejeita_rmax_invalido(rmax):
    with pytest.raises(ValueError, match="rmax"):
        ConjsFuzzy(rmax)


def test_rejeita_entrada_menor_que_kernel_sem_padding():
    with pytest.raises(ValueError, match="menor que o kernel"):
        FuzzyPooling(tamanho_kernel=3, padding=0)(torch.rand(1, 1, 2, 2))


@pytest.mark.parametrize("valor", [-0.01, 6.01, float("inf"), float("nan")])
def test_rejeita_valor_fora_do_universo_fuzzy(valor):
    entrada = torch.full((1, 1, 3, 3), valor)

    with pytest.raises(ValueError):
        FuzzyPooling(padding=0)(entrada)
