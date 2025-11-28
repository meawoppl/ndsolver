class Equation(dict):
    def __add__(self, o_dict):
        sum_dict = Equation()
        sum_dict.update(self)

        for key, val in o_dict.items():
            if key in sum_dict:
                sum_dict[key] += val
            else:
                sum_dict[key] = val
                
        sum_dict.zero_out()
        return sum_dict

    def __sub__(self, o_dict):
        sum_dict = Equation()
        sum_dict.update(self)
        
        for key, val in o_dict.items():
            if key in sum_dict:
                sum_dict[key] -= val
            else:
                sum_dict[key] = -val

        sum_dict.zero_out()
        return sum_dict


    def __mul__(self, mul):
        for key, val in self.items():
            self[key] *= mul

        self.zero_out()
        return self

    def __truediv__(self, div):
        if div == 0:
            raise ZeroDivisionError
        
        for key, val in self.items():
            self[key] /= div

        self.zero_out()
        return self

    def zero_out(self):
        for k in list(self.keys()):
            if self[k] == 0:
                del self[k]


    def __str__(self):
        print_list = []
        for key, val in self.items():
            print_list.append( "(#" + str(key) + ")*" + str(val))

        return " + ".join( print_list )

    def __getitem__(self, indx):
        return self.get(indx,0)

if __name__ == "__main__":
    A = Equation()
    B = Equation()

    A[3] = 1
    A[2] = 2
    A[1] = 3

    B[3] = 1
    B[1] = 3

    print("A", A)
    print("B", B)
    print("A + B", A + B)
    print("A - B", A - B)
    print("-B", Equation() - B)
    
