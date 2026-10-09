from reconhecimento_facial.features import decidir_acesso, distancia_cosseno, remover_acentos


def test_vetores_iguais_tem_distancia_zero():
    assert distancia_cosseno([1, 2, 3], [1, 2, 3]) < 1e-9


def test_vetores_ortogonais_tem_distancia_um():
    assert abs(distancia_cosseno([1, 0], [0, 1]) - 1.0) < 1e-9


def test_decisao_por_limiar():
    assert decidir_acesso(0.10, limiar=0.30) is True
    assert decidir_acesso(0.50, limiar=0.30) is False
    assert decidir_acesso(None) is False


def test_remover_acentos():
    assert remover_acentos("João Conceição") == "Joao Conceicao"
