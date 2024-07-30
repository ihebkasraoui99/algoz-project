import pytest
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig
from pathlib import Path

# Import functions from your main script
from algoz import initialization, get_datasets, get_algorithm, save_results

@pytest.fixture
def mock_config():
    return DictConfig({
        "output_path": "/fake/output/path",
        "mode": "train",
        "parameters": {"some_param": "value"}
    })

@pytest.fixture
def mock_configs(mock_config):
    mock_configs = MagicMock()
    mock_configs.algo = "some_algorithm"
    mock_configs.mode = "train"
    mock_configs.parameters = mock_config.parameters
    mock_configs.device = "cpu"
    return mock_configs

def test_initialization(mock_config):
    with patch('algoz.ConfigFactory.create_config') as mock_create_config, \
         patch('algoz.configure_logger') as mock_configure_logger, \
         patch('os.getcwd', return_value='/fake/cwd'), \
         patch('os.mkdir'), \
         patch('pathlib.Path.mkdir'):
        
        mock_create_config.return_value = MagicMock()
        configs, save_paths = initialization(mock_config)
        
        assert configs is not None
        assert "output" in save_paths

def test_get_datasets(mock_configs):
    with patch('algoz.DatasetFactory.create_dataset') as mock_create_dataset, \
         patch('algoz.save_data_distribution') as mock_save_data_distribution:
        
        mock_dataset = MagicMock()
        mock_create_dataset.return_value = mock_dataset
        
        datasets = get_datasets(mock_configs, {"output": Path("/fake/output/path")})
        
        assert "train" in datasets
        assert "test" in datasets

def test_get_algorithm(mock_configs):
    with patch('algoz.AlgorithmFactory.create_algorithm') as mock_create_algorithm:
        
        mock_algorithm = MagicMock()
        mock_create_algorithm.return_value = mock_algorithm
        
        algorithm = get_algorithm(mock_configs, {"train": MagicMock(), "test": MagicMock()}, {"output": Path("/fake/output/path")})
        
        assert algorithm is not None

def test_save_results(mock_configs):
    with patch('algoz.AvailableAlgorithm.get_type', return_value="classification") as mock_get_type, \
         patch('algoz.EvaluatorFactory.create_evaluator') as mock_create_evaluator, \
         patch('algoz.DashboardFactory.create_dashboard') as mock_create_dashboard, \
         patch('torch.no_grad'), \
         patch('pathlib.Path.mkdir'):
        
        mock_algorithm = MagicMock()
        mock_algorithm.model.module.eval = MagicMock()
        mock_algorithm.model.predict = MagicMock(return_value=[0])
        mock_algorithm.model.predict_proba = MagicMock(return_value=[0.5])
        
        mock_dataset = MagicMock()
        mock_dataset.ground_truth = [0]
        mock_dataset.config_data = {"gases_train": ["O2"], "gases_test": ["O2"]}
        
        mock_create_evaluator.return_value = MagicMock()
        mock_create_dashboard.return_value = MagicMock()
        
        evaluator, saved_dashboard_names = save_results(mock_configs, mock_algorithm, {"train": mock_dataset, "test": mock_dataset}, {"output": Path("./algoz-project/outputs")})
        
        assert evaluator is not None
        assert saved_dashboard_names is not None

# You can run the tests using the following command:
# pytest test_main.py
