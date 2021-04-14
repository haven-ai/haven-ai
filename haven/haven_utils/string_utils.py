import ast


def get_dict_from_str(string):
    if string is None:
        return string

    if string == "None":
        return None

    if string == "":
        return None

    return ast.literal_eval(string)


def get_list_from_str(string):
    if string is None:
        return string

    if string == "None":
        return None

    string = string.replace(" ", "").replace("]", "").replace("[", "").replace('"', "").replace("'", "")

    if string == "":
        return None

    return string.split(",")
