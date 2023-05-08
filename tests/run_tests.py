import pytest

print("\n===================================")
print("** Running dolfin tests **")
print("===================================\n")
pytest.main(["-v", "-m", "dolfin"])

print("\n===================================")
print("** Running smart model initialization tests **")
print("===================================\n")
pytest.main(["-v", "-m", "smart_model_init"])
