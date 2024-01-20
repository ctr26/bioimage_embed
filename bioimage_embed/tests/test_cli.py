# import os
# import pytest
# from ..hydra import main

# def test_main_creates_config():
#     # Arrange
#     config_path = "test_conf"
#     job_name = "test_app"

#     # Ensure the configuration directory does not exist initially
#     if os.path.exists(config_path):
#         os.rmdir(config_path)

#     # Act
#     main(config_path=config_path, job_name=job_name)

#     # Assert
#     assert os.path.exists(config_path), "Config directory was not created"
#     assert os.path.isfile(os.path.join(config_path, "config.yaml")), "Config file was not created"

#     # Clean up
#     os.remove(os.path.join(config_path, "config.yaml"))
#     os.rmdir(config_path)

# @pytest.mark.parametrize("config_path, job_name", [
#     ("conf", "test_app"),
#     ("another_conf", "another_job")
# ])
# def test_hydra_initializes(config_path, job_name):
#     # Act
#     main(config_path=config_path, job_name=job_name)

#     # Assert
#     # Here you can assert specifics about the cfg object if needed.
#     # Since main does not return anything, you might need to adjust
#     # the main function to return the cfg for more thorough testing.

#     # Clean up
#     if os.path.exists(config_path):
#         os.remove(os.path.join(config_path, "config.yaml"))
#         os.rmdir(config_path)
        