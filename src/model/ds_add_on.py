from pyds import MassFunction

class DSAddOn:

    def __init__(self, bpas):
        self.__bpas = bpas.copy()
        
    def set_mass(self):
        self.__masses = []
        for bpa in self.__bpas:
            mass = MassFunction({'n':bpa[0], 'na':bpa[1], 'a':bpa[2]}) 
            self.__masses.append(mass)
    
    def predict(self):
        assert len(self.__masses) >= 1
        bpa = self.__masses[0]
        if len(self.__masses) > 1:
            bpa = bpa.combine_conjunctive([m for m in self.__masses[1:]], normalization = True)
            
        return bpa, bpa.pl()