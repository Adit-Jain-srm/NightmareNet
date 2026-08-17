import unittest

from nightmarenet.evaluation.pareto_analysis import get_pareto_frontier


class TestParetoAnalysisUnit(unittest.TestCase):
    def test_pareto_dominance_detection(self):
        # A dominates B (higher robustness, lower latency, lower params)
        model_a = {"model": "A", "robustness": 0.9, "latency": 10.0, "params": 100}
        model_b = {"model": "B", "robustness": 0.8, "latency": 12.0, "params": 120}

        frontier = get_pareto_frontier([model_a, model_b])
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0]["model"], "A")

    def test_pareto_tradeoff_set(self):
        # A has higher robustness, B has lower latency
        model_a = {"model": "A", "robustness": 0.95, "latency": 20.0, "params": 100}
        model_b = {"model": "B", "robustness": 0.80, "latency": 5.0, "params": 100}

        frontier = get_pareto_frontier([model_a, model_b])
        self.assertEqual(len(frontier), 2)

    def test_empty_and_single_point(self):
        self.assertEqual(get_pareto_frontier([]), [])
        single = [{"model": "S", "robustness": 0.5, "latency": 1.0, "params": 10}]
        self.assertEqual(get_pareto_frontier(single), single)

    def test_identical_points(self):
        m1 = {"model": "M1", "robustness": 0.8, "latency": 10.0, "params": 50}
        m2 = {"model": "M2", "robustness": 0.8, "latency": 10.0, "params": 50}

        frontier = get_pareto_frontier([m1, m2])
        # Neither strictly dominates the other, so both remain
        self.assertEqual(len(frontier), 2)


if __name__ == "__main__":
    unittest.main()
