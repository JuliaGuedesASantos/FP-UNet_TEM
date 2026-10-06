import numpy as np
import hashlib
import gc
import os


# ============================================================
# CONFIGURAÇÕES
# ============================================================

N_DATASETS = 10

TRAIN_PREFIX = "datasets/graphene_defects/graphene_defects_dataset"
TEST_PREFIX = "datasets/graphene_defects/graphene_defects_test"

OUTPUT_TRAIN = "datasets/graphene_defects/graphene_defects_dataset_final.npz"
OUTPUT_TEST = "datasets/graphene_defects/graphene_defects_test_final.npz"

# Número de imagens copiadas por vez na segunda passagem.
# Quanto menor, menor o pico de memória.
BATCH_COPY = 10


# ============================================================
# FUNÇÃO:
# VERIFICAR SE UMA LABEL POSSUI MAIS DE UMA CLASSE
# ============================================================

def eh_multiclasse(label):
    """
    Retorna True se a imagem possui mais de uma classe.

    Exemplo:

        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]

    -> True

    Enquanto:

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

    -> False
    """

    return np.any(
        label != label.flat[0]
    )


# ============================================================
# PRIMEIRA PASSAGEM
# ============================================================

def analisar_datasets(
    prefixo,
    n_datasets=10
):
    """
    Primeira passagem pelos arquivos.

    Não cria o dataset final.

    Apenas determina:
        - quais imagens possuem múltiplas classes;
        - quais imagens são repetidas;
        - quais índices devem ser mantidos.

    A comparação de duplicatas é GLOBAL entre todos os
    datasets do conjunto.

    Exemplo:

        dataset0 -> imagem A
        dataset1 -> imagem B
        dataset2 -> imagem A

    A imagem A do dataset2 será removida.
    """

    # --------------------------------------------------------
    # Hashes de todas as imagens já encontradas.
    #
    # IMPORTANTE:
    # Este set permanece durante os 10 datasets.
    # --------------------------------------------------------

    vistos = set()

    # Índices válidos de cada dataset
    indices_validos = []

    # Contadores
    total = 0
    removidas_classe = 0
    removidas_repetidas = 0

    # Informações dos arrays
    dtype_image = None
    dtype_label = None

    shape_image = None
    shape_label = None


    print()
    print("=" * 70)
    print("PRIMEIRA PASSAGEM")
    print("=" * 70)


    # ========================================================
    # PROCESSA CADA DATASET
    # ========================================================

    for dataset_id in range(n_datasets):

        nome = f"{prefixo}{dataset_id}.npz"

        print()
        print(f"Processando: {nome}", flush=True)


        # ----------------------------------------------------
        # Abre somente o arquivo atual
        # ----------------------------------------------------

        with np.load(nome) as arquivo:

            images = arquivo["images"]
            labels = arquivo["labels"]


            # ------------------------------------------------
            # Informações estruturais
            # ------------------------------------------------

            if dtype_image is None:

                dtype_image = images.dtype
                dtype_label = labels.dtype

                shape_image = images.shape[1:]
                shape_label = labels.shape[1:]


            print(
                f"  Images: {images.shape}",
                flush=True
            )

            print(
                f"  Labels: {labels.shape}",
                flush=True
            )

            print(
                f"  Image dtype: {images.dtype}",
                flush=True
            )

            print(
                f"  Label dtype: {labels.dtype}",
                flush=True
            )


            # ------------------------------------------------
            # Índices que sobreviverão neste dataset
            # ------------------------------------------------

            indices_dataset = []


            # =================================================
            # ANALISA CADA IMAGEM
            # =================================================

            for j in range(len(images)):

                total += 1


                # ---------------------------------------------
                # 1. REMOVER IMAGENS DE CLASSE ÚNICA
                # ---------------------------------------------

                if not eh_multiclasse(labels[j]):

                    removidas_classe += 1

                    continue


                # ---------------------------------------------
                # 2. HASH DA IMAGEM
                # ---------------------------------------------

                h = hashlib.sha1(
                    images[j].tobytes()
                ).digest()


                # ---------------------------------------------
                # 3. VERIFICAR SE JÁ FOI VISTA
                # ---------------------------------------------

                if h in vistos:

                    removidas_repetidas += 1

                    continue


                # ---------------------------------------------
                # 4. IMAGEM VÁLIDA
                # ---------------------------------------------

                vistos.add(h)

                indices_dataset.append(j)


        # ====================================================
        # Guarda os índices deste dataset
        # ====================================================

        indices_validos.append(
            np.asarray(
                indices_dataset,
                dtype=np.int32
            )
        )


        print(
            f"  Imagens válidas: "
            f"{len(indices_dataset)}",
            flush=True
        )


        print(
            f"  Imagens removidas por classe única: "
            f"{removidas_classe}",
            flush=True
        )


        print(
            f"  Imagens repetidas removidas: "
            f"{removidas_repetidas}",
            flush=True
        )


        # ----------------------------------------------------
        # Libera explicitamente a memória
        # ----------------------------------------------------

        del images
        del labels

        gc.collect()


    # ========================================================
    # TOTAL FINAL
    # ========================================================

    total_validas = sum(
        len(indices)
        for indices in indices_validos
    )


    print()
    print("=" * 70)
    print("RESULTADO DA PRIMEIRA PASSAGEM")
    print("=" * 70)

    print(
        f"Total original:          {total}"
    )

    print(
        f"Removidas classe única:  {removidas_classe}"
    )

    print(
        f"Removidas repetidas:     {removidas_repetidas}"
    )

    print(
        f"Total final:             {total_validas}"
    )

    print("=" * 70)


    return (
        indices_validos,
        total_validas,
        dtype_image,
        dtype_label,
        shape_image,
        shape_label
    )


# ============================================================
# SEGUNDA PASSAGEM
# CONSTRUÇÃO DO DATASET USANDO MEMMAP
# ============================================================

def construir_memmap(
    prefixo,
    indices_validos,
    total_validas,
    dtype_image,
    dtype_label,
    shape_image,
    shape_label,
    nome_temp,
    n_datasets=10
):
    """
    Segunda passagem.

    Cria os arrays finais diretamente em arquivos temporários
    no disco através de np.memmap.

    Assim, o dataset final não precisa ocupar toda a RAM.
    """

    print()
    print("=" * 70)
    print("SEGUNDA PASSAGEM - MEMMAP")
    print("=" * 70)


    # ========================================================
    # NOMES DOS ARQUIVOS TEMPORÁRIOS
    # ========================================================

    images_path = (
        f"{nome_temp}_images.dat"
    )

    labels_path = (
        f"{nome_temp}_labels.dat"
    )


    # ========================================================
    # CRIA MEMMAP DAS IMAGENS
    # ========================================================

    print()
    print(
        "Criando memmap das imagens...",
        flush=True
    )


    images_final = np.memmap(
        images_path,
        dtype=dtype_image,
        mode="w+",
        shape=(
            total_validas,
            *shape_image
        )
    )


    # ========================================================
    # CRIA MEMMAP DAS LABELS
    # ========================================================

    print(
        "Criando memmap das labels...",
        flush=True
    )


    labels_final = np.memmap(
        labels_path,
        dtype=dtype_label,
        mode="w+",
        shape=(
            total_validas,
            *shape_label
        )
    )


    # ========================================================
    # POSIÇÃO ATUAL NO DATASET FINAL
    # ========================================================

    pos = 0


    # ========================================================
    # PROCESSA CADA DATASET
    # ========================================================

    for dataset_id in range(n_datasets):

        nome = f"{prefixo}{dataset_id}.npz"

        idx = indices_validos[dataset_id]

        n = len(idx)


        print()
        print(
            f"Copiando: {nome}",
            flush=True
        )

        print(
            f"  Imagens válidas: {n}",
            flush=True
        )


        # ----------------------------------------------------
        # Se não houver imagens válidas
        # ----------------------------------------------------

        if n == 0:

            print(
                "  Nenhuma imagem para copiar.",
                flush=True
            )

            continue


        # ----------------------------------------------------
        # Abre somente o dataset atual
        # ----------------------------------------------------

        with np.load(nome) as arquivo:

            images = arquivo["images"]
            labels = arquivo["labels"]


            # =================================================
            # COPIA EM PEQUENOS BLOCOS
            # =================================================

            for start in range(
                0,
                n,
                BATCH_COPY
            ):

                end = min(
                    start + BATCH_COPY,
                    n
                )


                idx_batch = idx[
                    start:end
                ]


                n_batch = len(
                    idx_batch
                )


                # ---------------------------------------------
                # Copia imagens
                # ---------------------------------------------

                images_final[
                    pos:pos + n_batch
                ] = images[
                    idx_batch
                ]


                # ---------------------------------------------
                # Copia labels
                # ---------------------------------------------

                labels_final[
                    pos:pos + n_batch
                ] = labels[
                    idx_batch
                ]


                # ---------------------------------------------
                # Atualiza posição
                # ---------------------------------------------

                pos += n_batch


                # ---------------------------------------------
                # Libera temporários do batch
                # ---------------------------------------------

                del idx_batch

                gc.collect()


                print(
                    f"\r  Copiado: "
                    f"{pos}/{total_validas}",
                    end="",
                    flush=True
                )


        print()


        # ----------------------------------------------------
        # Força gravação no disco
        # ----------------------------------------------------

        images_final.flush()
        labels_final.flush()


        # ----------------------------------------------------
        # Libera dataset atual
        # ----------------------------------------------------

        del images
        del labels

        gc.collect()


    # ========================================================
    # GARANTE QUE TUDO FOI GRAVADO
    # ========================================================

    images_final.flush()
    labels_final.flush()


    print()
    print(
        "Memmaps construídos com sucesso.",
        flush=True
    )


    print(
        f"Images: {images_final.shape}",
        flush=True
    )

    print(
        f"Labels: {labels_final.shape}",
        flush=True
    )


    return (
        images_final,
        labels_final,
        images_path,
        labels_path
    )


# ============================================================
# SALVAR COMO NPZ
# ============================================================

def salvar_npz(
    images_memmap,
    labels_memmap,
    output_file
):
    """
    Salva os memmaps como um único arquivo .npz.

    Os nomes das chaves serão:

        image
        labels
    """

    print()
    print("=" * 70)
    print("SALVANDO NPZ")
    print("=" * 70)

    print(
        f"Arquivo: {output_file}",
        flush=True
    )


    # --------------------------------------------------------
    # Salva
    # --------------------------------------------------------

    np.savez(
        output_file,
        image=images_memmap,
        labels=labels_memmap
    )


    print(
        "NPZ salvo com sucesso!",
        flush=True
    )


    # --------------------------------------------------------
    # Verifica tamanho
    # --------------------------------------------------------

    if os.path.exists(output_file):

        tamanho_gb = (
            os.path.getsize(output_file)
            / (1024 ** 3)
        )

        print(
            f"Tamanho do arquivo: "
            f"{tamanho_gb:.3f} GB",
            flush=True
        )


# ============================================================
# PROCESSAR UM CONJUNTO COMPLETO
# ============================================================

def processar_conjunto(
    prefixo,
    output_file,
    nome_temp,
    n_datasets=10
):
    """
    Executa:

        1. primeira passagem;
        2. segunda passagem;
        3. criação do .npz;
        4. remoção dos arquivos temporários.
    """

    # ========================================================
    # PRIMEIRA PASSAGEM
    # ========================================================

    (
        indices_validos,
        total_validas,
        dtype_image,
        dtype_label,
        shape_image,
        shape_label
    ) = analisar_datasets(
        prefixo=prefixo,
        n_datasets=n_datasets
    )


    # ========================================================
    # SEGUNDA PASSAGEM
    # ========================================================

    (
        images_memmap,
        labels_memmap,
        images_path,
        labels_path
    ) = construir_memmap(

        prefixo=prefixo,

        indices_validos=indices_validos,

        total_validas=total_validas,

        dtype_image=dtype_image,

        dtype_label=dtype_label,

        shape_image=shape_image,

        shape_label=shape_label,

        nome_temp=nome_temp,

        n_datasets=n_datasets
    )


    # ========================================================
    # LIBERA ÍNDICES
    # ========================================================

    del indices_validos

    gc.collect()


    # ========================================================
    # SALVA NPZ
    # ========================================================

    salvar_npz(
        images_memmap,
        labels_memmap,
        output_file
    )


    # ========================================================
    # FECHA OS MEMMAPS
    # ========================================================

    del images_memmap
    del labels_memmap

    gc.collect()


    # ========================================================
    # REMOVE ARQUIVOS TEMPORÁRIOS
    # ========================================================

    print()
    print(
        "Removendo arquivos temporários...",
        flush=True
    )


    if os.path.exists(images_path):

        os.remove(images_path)


    if os.path.exists(labels_path):

        os.remove(labels_path)


    gc.collect()


    print(
        "Arquivos temporários removidos.",
        flush=True
    )


    print()
    print("=" * 70)
    print(
        f"PROCESSAMENTO DE {prefixo} CONCLUÍDO"
    )
    print("=" * 70)


# ============================================================
# PROCESSAMENTO DO DATASET DE TREINAMENTO
# ============================================================

print()
print("#" * 70)
print("# DATASET DE TREINAMENTO")
print("#" * 70)


processar_conjunto(

    prefixo=TRAIN_PREFIX,

    output_file=OUTPUT_TRAIN,

    nome_temp="temp_train",

    n_datasets=N_DATASETS
)


# ============================================================
# GARBAGE COLLECTION
# ============================================================

gc.collect()


# ============================================================
# PROCESSAMENTO DO DATASET DE TESTE
# ============================================================

print()
print("#" * 70)
print("# DATASET DE TESTE")
print("#" * 70)


processar_conjunto(

    prefixo=TEST_PREFIX,

    output_file=OUTPUT_TEST,

    nome_temp="temp_test",

    n_datasets=N_DATASETS
)


# ============================================================
# FINAL
# ============================================================

print()
print("#" * 70)
print("# PROCESSAMENTO COMPLETO")
print("#" * 70)

print(
    f"Dataset de treinamento: {OUTPUT_TRAIN}"
)

print(
    f"Dataset de teste:       {OUTPUT_TEST}"
)

print(
    "\nConcluído!"
)
