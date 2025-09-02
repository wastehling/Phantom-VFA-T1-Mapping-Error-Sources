import importlib


class DataLoaderCreator:
    """
    class to import and create data loading functions
    """
    @staticmethod
    def create(class_name):
        """
        creates a class object from a string
        Args:
            class_name: name of the class to create

        Returns:
            class object

        """
        package, class_name = class_name.rsplit('.', 1)
        model_module = importlib.import_module(package)
        class_object = getattr(model_module, class_name)
        return class_object
