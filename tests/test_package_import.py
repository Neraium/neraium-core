import neraium_core


def test_package_exports():
    assert hasattr(neraium_core, "__all__")
