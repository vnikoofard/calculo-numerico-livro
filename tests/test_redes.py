import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cnlivro.redes import Padronizador, mlp, treinar


def test_mlp_forma_e_padronizador():
    rede = mlp(entradas=2, saidas=3, largura=8, camadas=2)
    assert rede(torch.zeros(5, 2)).shape == (5, 3)
    dados = np.array([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]])
    p = Padronizador(dados)
    z = p(dados)
    assert torch.allclose(z.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(p.inverso(z), torch.as_tensor(dados, dtype=z.dtype), atol=1e-5)


def test_treino_reduz_perda_e_ajusta():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 64)[:, None]
    y = torch.sin(3 * x)
    r = treinar(mlp(largura=16, camadas=2), x, y, epocas=1500, lr=1e-2, x_val=x[::3], y_val=y[::3],
                instantes=[1, 1500], callback=lambda m: m(x).numpy().copy())
    assert r.historico[-1] < 1e-3 * r.historico[0]
    assert len(r.historico_validacao) == 1500
    assert set(r.instantaneos) == {1, 1500}
