import os
import unittest
import importlib.util

def discover_tests(start_dir):
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.test.py'):
                file_path = os.path.join(root, file)
                module_name = file[:-3]  # Remove the '.py' extension
                
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                module_tests = test_loader.loadTestsFromModule(module)
                test_suite.addTests(module_tests)

    return test_suite

if __name__ == '__main__':
    test_suite = discover_tests('.')
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)
