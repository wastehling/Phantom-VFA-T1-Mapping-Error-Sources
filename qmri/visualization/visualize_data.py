import logging

from qmri.visualization.VisualizationCreator import VisualizationCreator


def visualize_data(data, target_folder, visualization_parameters):
    """
    wrapper function to construct visualization objects according to visualization parameters
    Args:
        data: dict containing torchio Subjects, organized per type, application, study and subject ID
        target_folder: path where to store plots
        visualization_parameters: dict containing visualization parameters
    """
    for type in data.keys():
        type_folder = target_folder + '/plots/' + type
        for application in data[type].keys():
            application_folder = type_folder + '/' + application
            for study in data[type][application].keys():
                study_folder = application_folder + '/' + study
                for subject_id, subject in enumerate(data[type][application][study]):
                    subject_folder = study_folder + '/' + str(subject_id)

                    for visualization_module, module_parameters in visualization_parameters.items():
                        visualizer = VisualizationCreator.create(visualization_module)
                        function_name = visualization_module.rsplit('.', 1)[1]
                        visualizer(subject, module_parameters, subject_folder, name=function_name)

                        try:
                            visualizer(subject, module_parameters, subject_folder, name=function_name)
                        except Exception as e:
                            logging.info(f'error: {str(e)}, {function_name} plots not possible')
                            pass


