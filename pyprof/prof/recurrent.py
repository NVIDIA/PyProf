# Author: Aditya Agrawal
# Email: adityaa@nvidia.com

# This file supports torch.(RNN, LSTM, GRU)
# The file recurrentCell.py supports torch.(RNN, LSTM, GRU)Cell

from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase
import math

def str2Dict(modMarker):

    cellType = modMarker['mod']
    assert cellType in ["LSTM", "GRU"]
    modArgs = modMarker['strRepr'].replace(" ", "").split(",")

    d = {}
    d['cell_type'] = cellType
    d['input_size'] = eval(modArgs[0])
    d['hidden_size'] = eval(modArgs[1])
    d['num_layers'] = 1
    d['bias'] = True
    d['batch_first'] = False
    d['dropout'] = 0
    d['bidirectional'] = False

    for i in range(2, len(modArgs)):
        param = modArgs[i]
        assert ("=" in param)
        k,v = param.split("=")
        d[k] = eval(v)

    return d

def fusionSteps(direc, b):
    #b is the batch size
    #this is from cuDNN heuristics
    assert (direc in ['fprop', 'bprop'])

    if direc == "fprop":
        if b <=1:
            return 32
        elif b <= 2:
            return 16
        elif b <= 4:
            return 16
        elif b <= 8:
            return 16
        elif b <= 16:
            return 8
        elif b <= 48:
            return 4
        else:
            return 2
    else:
        if b <=1:
            return 32
        elif b <= 2:
            return 16
        elif b <= 4:
            return 16
        elif b <= 8:
            return 16
        elif b <= 16:
            return 8
        elif b <= 96:
            return 4
        else:
            return 2

def ctaTile(name):
    name = name.split("_")
    name = list(filter(lambda x : "x" in x, name))
    name = list(filter(lambda x : "slice" not in x, name))
    assert(len(name) == 1)
    name = name[0].split("x")
    assert(len(name) == 2)
    name = list(map(int, name))
    return name[0], name[1]

def hasTileSize(name):
    if ("sgemm" in name) or ("884gemm" in name) or ("hgemm" in name):
        return True
    else:
        return False

class RNNBase(OperatorLayerBase):

    def __init__(self, d):

        assert(len(d.modMarker))
        modMarker = eval(d.modMarker[0])
        modArgs = str2Dict(modMarker)

        argMarker = eval(d.argMarker[0])
        mod = argMarker['mod']
        op = argMarker['op']

        #Might need to be fixed
        self.mod_ = mod
        self.op_ = op

        #Determine the batch size and sequence length
        #This depends on if the input is packed or not
        assert (len(argMarker))
        args = argMarker['args']
        arg0 = args[0]
        len0 = len(arg0['shape'])
        assert (len0 == 2) or (len0 == 3)
        packed = (len0 == 2)

        b = 0
        l = 0
        x = 0

        if packed:
            assert (len(argMarker) >= 2)
            arg1 = args[1]
            l = arg1['shape'][0]
            x = arg0['shape'][1]
            b = math.ceil(arg0['shape'][0] / l)
            #b could be slightly less than a power of 2 (because of packing)
            #Check the fractional part
            frac,integer = math.modf(math.log(b,2))
            if frac > 0.9:
                b = pow(2, integer+1)
            b = int(b)
        else:
            if modArgs['batch_first']:
                b,l,x = arg0['shape']
            else:
                l,b,x = arg0['shape']

        assert x == modArgs['input_size']
        h = modArgs['hidden_size']

        self.name = d.name
        self.dir = d.dir
        self.sub = d.sub
        self.grid = d.grid

        self.cell = modArgs['cell_type']
        self.b = b
        self.seqL = l
        self.L = modArgs['num_layers']
        self.h = h
        self.x = x
        self.biDir = modArgs['bidirectional']
        self.type = arg0['dtype']

        #params, sol, limiter, flops, bytes

    def params(self):
        p,_,_ = self.bar()
        if p is None:
            p = OrderedDict([('cell', self.cell), ('B', self.b), ('seqL', self.seqL), ('L', self.L), ('H', self.h), ('X', self.x), ('biDir', self.biDir), ('type', self.type)])
            return p
        else:
            return p

    def tc(self):
        if "gemm" in self.name:
            return 1 if "884gemm" in self.name else 0
        else:
            return "-"

    def op(self):
        return self.op_

    def mod(self):
        return self.mod_

    def bytes(self):
        _,_,b = self.bar()
        return b

    def flops(self):
        _,f,_ = self.bar()
        return f

    def bar(self):
        cell = self.cell
        B = self.b
        H = self.h
        L = self.L
        X = self.x
        seqL = self.seqL
        biDir = self.biDir
        t = self.type

        flops = 0
        bytes = 0
        params = None

        if (dir == "fprop"):
            if name == "cudnn::detail::dropout_fp":
                elems = B*H
                bytes = elems*Utility.typeToBytes(t)
                params = OrderedDict([('B',B),('H',H),('type',t)])

            elif name == "{}_elementWise_fp".format(cell):
                if cell == "LSTM":
                    # input: 2*(i,f,g,o),c_(t-1)
                    # output: c_t, h_t
                    elems = 11*B*H
                elif cell == "GRU":
                    # input: 2*(r,z,n),h_(t-1)
                    # output: h_t
                    elems = 8*B*H
                else:
                    assert False

                bytes = elems*Utility.typeToBytes(t)
                params = OrderedDict([('B',B),('H',H),('type',t)])

            elif ("gemm" in name) and hasTileSize(name):    #gemm
                #Get cta tile size
                tileX, tileY = ctaTile(name)
                #Get grid dimensions
                grid = grid.split(",")
                gridX,gridY,gridZ = map(lambda x : int(x), grid)

                gemmM = tileX * gridX
                gemmN = tileY * gridY
                timesteps = fusionSteps("fprop", B)

                #assert(gemmM in [H, 3*H, 4*H]) #RNN or GRU or LSTM
                #assert(gemmK in [H, 2H, X]) #depending on the layer and if bidirectional/unidirectional
                #Might not always be true because of quantization etc.

                if (gemmN == B):    #recurrent gemm
                    gemmK = H
                    params = OrderedDict([('gemm','fprop'),('M',gemmM),('N',gemmN),('K',gemmK),('type',t)])
                elif (gemmN == B*timesteps):    #layer gemm
                    gemmK = X    #could be H, X, or 2H. Refine TODO
                    params = OrderedDict([('gemm','fprop'),('M',gemmM),('N',gemmN),('K',gemmK),('type',t)])
                else:
                    pass

        elif (dir == "bprop"):
            if name == "cudnn::detail::dropout_bp":
                elems = B*H
                bytes = elems*Utility.typeToBytes(t)
                params = OrderedDict([('B',B),('H',H),('type',t)])
            elif ("gemm" in name) and hasTileSize(name):    #gemm
                #There are four cases
                # recurrent dgrad: M = H, N = B, K = 4H, NN
                # layer dgrad:     M = X, N = B.timesteps, K = 4H, NN

                # recurrent wgrad: M = H, N = 4H, K = B.seqL, NT
                # layer wgrad:     M = X, N = 4H, K = B.seqL, NT

                #Get cta tile size
                tileX, tileY = ctaTile(name)
                #Get grid dimensions
                grid = grid.split(",")
                gridX,gridY,gridZ = map(lambda x : int(x), grid)

                gemmM = tileX * gridX
                gemmN = tileY * gridY
                timesteps = fusionSteps("bprop", B)

                #print(name, gridX, gridY, gridZ, gemmM, gemmN)
                if name[-3:] == "_nn": # dgrad
                    #assert (gemmM in [H, X])
                    #assert (gemmN in [B, B*timesteps])
                    #Might not always be true because of quantization.

                    gemmK = H
                    if cell == "LSTM":
                        gemmK *= 4
                    elif cell == "GRU":
                        gemmK *= 3

                    params = OrderedDict([('gemm','dgrad'),('M',gemmM),('N',gemmN),('K',gemmK),('type',t)])

                elif name[-3:] == "_nt": #wgrad
                    #assert (gemmM in [H, X])
                    #assert (gemmN in [H, 3*H, 4*H])
                    #Might not always be true because of quantization.
                    #If gemmM == H, implies recurrent wgrad
                    #If gemmM == X, implies layer wgrad
                    #If H==X, cannot distinguish between recurrent and layer wgrad

                    gemmK = B*seqL
                    params = OrderedDict([('gemm','wgrad'),('M',gemmM),('N',gemmN),('K',gemmK),('type',t)])

                else:
                    pass

            else:
                pass

        return params, flops, bytes
