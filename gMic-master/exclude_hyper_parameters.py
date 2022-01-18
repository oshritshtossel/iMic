import json
import os


def get_hyper_parameters_as_dict(params_file):
    f = open(params_file, "r")
    parames_dict = {}
    c = 0
    for line in f:
        if line == "\n":
            c += 1
        if c == 2:
            break

    for line in f:
        x = line.split(",")
        # x = x[2:-1]
        x = [i.replace("\n", "") for i in x]
        try:
            parames_dict[x[0]] = float(x[1])
        except:
            try:
                parames_dict[x[0]] = int(x[1])
            except:
                parames_dict[x[0]] = x[1]
    return parames_dict


if __name__ == '__main__':
    for dirpath, dirnames, filenames in os.walk(os.path.join("reported_results")):
        for file in filenames:
            params_dict = get_hyper_parameters_as_dict(os.path.join("reported_results", file))
            file_name = file.split("_val_")[0]
            with open(file_name + ".json", 'w') as fp:
                json.dump(dict(sorted(params_dict.items())), fp)

