import pytest

print("\n===================================")
print("** Running dolfin tests **")
print("===================================\n")
pytest.main(["-v", "-m", "dolfin"])

print("\n===================================")
print("** Running stubs model setup tests **")
print("===================================\n")
pytest.main(["-v", "-m", "stubs_model_setup"])

print("\n===================================")
print("** Running stubs model initialization tests **")
print("===================================\n")
pytest.main(["-v", "-m", "stubs_model_init"])
