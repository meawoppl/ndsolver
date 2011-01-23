import os

def make_directory(dur):
    if not os.path.isdir(dur):
        os.mkdir(dur)

def path_to_dict(path):
    return_dict = {}
    for f in os.listdir(path):
        temp_path = os.path.join(path, f)
        if os.path.isdir(temp_path):
            results = path_to_dict(temp_path)
        else:
            results = open(temp_path,"rb").read()
        trash, temp_path = os.path.split(temp_path)
        return_dict[temp_path] = results

    return return_dict

def dict_to_path(dictionary, landing=os.getcwd()):
    for name, data in dictionary.iteritems():
        new_path = os.path.join(landing, name)
        if type(data) == dict:
            make_directory(new_path)
            dict_to_path(data, landing=new_path)
        else:
            open(new_path, "wb").write(data)
