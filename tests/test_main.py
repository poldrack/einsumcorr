import pytest
import sys
from io import StringIO
from unittest.mock import patch
import numpy as np
from einsumcorr import main


def test_main_smoke():
    """Smoke test for main function - ensure it can be called."""
    # Test that main function exists and can be called
    with patch('sys.argv', ['einsumcorr', '--help']):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # --help should exit with code 0
        assert exc_info.value.code in [0, None]


def test_main_no_args():
    """Test main function with no arguments shows help."""
    with patch('sys.argv', ['einsumcorr']):
        with pytest.raises(SystemExit):
            main()


def test_main_single_file():
    """Test main function with single input file."""
    # Create a temporary test file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write test data
        data = np.random.randn(10, 3)
        np.savetxt(f, data, delimiter=',')
        temp_file = f.name
    
    try:
        with patch('sys.argv', ['einsumcorr', temp_file]):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                # Check that some output was produced
                assert len(output) > 0
    finally:
        os.unlink(temp_file)


def test_main_two_files():
    """Test main function with two input files."""
    import tempfile
    import os
    
    # Create two temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
        data1 = np.random.randn(10, 3)
        np.savetxt(f1, data1, delimiter=',')
        temp_file1 = f1.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
        data2 = np.random.randn(10, 2)
        np.savetxt(f2, data2, delimiter=',')
        temp_file2 = f2.name
    
    try:
        with patch('sys.argv', ['einsumcorr', temp_file1, temp_file2]):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                # Check that some output was produced
                assert len(output) > 0
    finally:
        os.unlink(temp_file1)
        os.unlink(temp_file2)


def test_main_invalid_file():
    """Test main function with invalid file path."""
    with patch('sys.argv', ['einsumcorr', 'nonexistent_file.csv']):
        with pytest.raises(SystemExit):
            main()


def test_main_output_file():
    """Test main function with output file option."""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_in:
        data = np.random.randn(10, 3)
        np.savetxt(f_in, data, delimiter=',')
        temp_input = f_in.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_out:
        temp_output = f_out.name
    
    try:
        with patch('sys.argv', ['einsumcorr', temp_input, '--output', temp_output]):
            main()
            # Check that output file was created and has content
            assert os.path.exists(temp_output)
            assert os.path.getsize(temp_output) > 0
    finally:
        os.unlink(temp_input)
        if os.path.exists(temp_output):
            os.unlink(temp_output)