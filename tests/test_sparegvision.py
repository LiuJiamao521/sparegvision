import numpy as np
import torch

from sparegvision.baselines import best_single_enhancer, global_nonnegative_elastic_net
from sparegvision.metrics import prediction_metrics
from sparegvision.model import SpatialAttributionSetNetwork
from sparegvision.simulation import simulate_spatial_decomposition
from sparegvision.spatial_cv import contiguous_grid_blocks, spatial_folds


def test_spatial_folds_do_not_overlap():
    yy, xx = np.mgrid[:10, :12]
    coords = np.c_[xx.ravel(), yy.ravel()]
    labels = contiguous_grid_blocks(coords, n_folds=4)
    for train, test in spatial_folds(labels):
        assert not np.intersect1d(train, test).size
        assert len(train) + len(test) == len(coords)


def test_complementary_simulation_has_known_domains():
    sim = simulate_spatial_decomposition("complementary", shape=(24, 32), seed=3)
    assert sim.enhancer_maps.shape == (4, 24, 32)
    assert np.allclose(sim.attribution.sum(0)[sim.tissue_mask], 1.0)
    assert np.any(sim.attribution[0] > 0) and np.any(sim.attribution[1] > 0)


def test_multi_linear_beats_single_on_complementary_holdout():
    sim = simulate_spatial_decomposition("complementary", shape=(30, 40), noise=0.01, seed=4)
    train = sim.tissue_mask & (np.indices(sim.tissue_mask.shape)[0] % 3 != 0)
    test = sim.tissue_mask & ~train
    ps, ys, _ = best_single_enhancer(sim.enhancer_maps, sim.gene_map, train, test)
    pm, ym, _ = global_nonnegative_elastic_net(sim.enhancer_maps, sim.gene_map, train, test)
    assert prediction_metrics(ym, pm)["mse"] < prediction_metrics(ys, ps)["mse"]


def test_model_outputs_and_permutation_equivariance():
    torch.manual_seed(2)
    model = SpatialAttributionSetNetwork(hidden_dim=16, attention_heads=4, set_layers=1, dropout=0)
    model.eval()
    enh = torch.rand(2, 3, 1, 12, 10)
    context = torch.rand(2, 3, 12, 10)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    with torch.no_grad():
        out = model(enh, context, mask)
        permutation = torch.tensor([2, 0, 1])
        inv = torch.argsort(permutation)
        out_p = model(enh[:, permutation], context, mask[:, permutation])
    assert out["gene_prediction"].shape == (2, 1, 12, 10)
    assert out["attribution_maps"].shape == (2, 3, 12, 10)
    assert torch.all(out["attribution_maps"] >= 0)
    assert torch.all(out["background_map"] >= 0)
    assert torch.allclose(out["gene_prediction"], out_p["gene_prediction"], atol=1e-5)
    assert torch.allclose(out["attribution_maps"], out_p["attribution_maps"][:, inv], atol=1e-5)
    assert torch.all(out["attribution_maps"][1, 2] == 0)



def test_mixed_complexity_simulation_contains_expected_enhancer_classes():
    from sparegvision.complexity_simulation import simulate_mixed_regulatory_complexity
    sim = simulate_mixed_regulatory_complexity(seed=11)
    assert sim.enhancer_maps.shape[0] == 20
    assert sim.enhancer_classes.count("core_concordant") == 10
    assert sim.enhancer_classes.count("regional_specific") == 3
    assert sim.enhancer_classes.count("regional_redundant") == 2
    assert sim.enhancer_classes.count("unsupported") == 5


def test_complexity_evidence_recovers_regional_enhancers():
    from sparegvision.complexity import enhancer_evidence_table
    from sparegvision.complexity_simulation import simulate_mixed_regulatory_complexity
    sim = simulate_mixed_regulatory_complexity(seed=12)
    evidence = enhancer_evidence_table(sim.enhancer_maps, sim.gene_map, sim.tissue_mask,
        truth_classes=sim.enhancer_classes, truth_domains=sim.enhancer_domains, seed=12)
    truth = evidence["truth_class"].isin(["regional_specific", "regional_redundant"])
    predicted = evidence["predicted_class"] == "regional_candidate"
    assert int((truth & predicted).sum()) == 5
    assert int((~truth & predicted).sum()) == 0
