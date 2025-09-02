import csv


def save_loss_history(path, loss_dict):
    """
    save loss history
    Args:
        path: path where to save loss history
        loss_dict: loss dict to save

    """

    loss_hist_path = path + '/metric_history.csv'
    with open(loss_hist_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        for key, value in loss_dict.items():
            if isinstance(value, list):
                if not isinstance(value[0], str):
                    value.insert(0, key)
                wr.writerow(value)

            # if value is a dict, unpack it
            elif isinstance(value, dict):

                # unpack per study
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, list):
                        sub_value = [sub_value]

                    # if list of lists, unpack again
                    if isinstance(sub_value, list) and all(isinstance(item, list) for item in sub_value):
                        for item in sub_value:
                            if len(item) == 1:
                                item.insert(0, key + '_' + sub_key)
                            wr.writerow(item)

                    # if not list of lists, write row
                    else:
                        if len(sub_value) == 1:
                            sub_value.insert(0, key + '_' + sub_key)
                        wr.writerow(sub_value)


def save_metric_dict(metric_dict, target_folder):
    """
    save metric dict
    Args:
        metric_dict: metric dict to save
        target_folder: target folder to save metric dict

    """
    metric_dict_patth = target_folder + '/metrics.csv'
    with open(metric_dict_patth, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        if metric_dict:
            for type in metric_dict.keys():
                for application in metric_dict[type].keys():
                    for study in metric_dict[type][application].keys():
                        for parameter_name in metric_dict[type][application][study].keys():
                            for metric_name, values in metric_dict[type][application][study][parameter_name].items():
                                if not isinstance(values, list):
                                    values = [values]
                                values.insert(0, metric_name)
                                values.insert(0, parameter_name)
                                values.insert(0, study)
                                values.insert(0, application)
                                values.insert(0, type)
                                wr.writerow(values)