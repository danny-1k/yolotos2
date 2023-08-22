import torch


class Binning:
    def __init__(self, bins):
        self.bins = torch.linspace(0, 1, bins)
        self.tokens = ["SOS", "PAD"] + self.bins.tolist()
        self.n_tokens = len(self.tokens)
        self.token_ix = {t: i for i, t in enumerate(self.tokens)}
        self.ix_token = {i:t for i, t in enumerate(self.tokens)}
        
    def encode(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        indices = torch.floor((x/self.bins[1])+.5) # put it in closest bin
        indices = indices.long() + 2 # for SOS & PAD
        return indices
    
    def decode(self, x):
        return self.bins[x]

class Classes:
    def __init__(self, classes):
        self.tokens = ["SOS", "PAD"] + classes
        self.n_tokens = len(self.tokens)
        self.token_ix = {t: i for i, t in enumerate(self.tokens)}
        self.ix_token = {i:t for i, t in enumerate(self.tokens)}

    def encode(self, x):
        indices = [self.token_ix[t] for t in x]
        return indices
    
    def decode(self, tokens):
        # tokens of shape (N)
        decoded = []
        for n in tokens:
            decoded.append(self.ix_token[n])
        return decoded

class Encoder:
    def __init__(self, classes, binning):
        self.classes = classes
        self.binning = binning
        self.n_tokens = classes.n_tokens + binning.n_tokens

    def encode(self, classes, x, y, w, h):
        classes = self.classes.encode(classes)
        x = self.binning.encode(x)
        y = self.binning.encode(y)
        w = self.binning.encode(w)
        h = self.binning.encode(h)

        return classes, x, y, w, h

    def decode(self, classes, x, y, w, h):
        # tokens of shape (N)

        decoded_classes = self.classes.decode(classes)
        decoded_x = self.x.decode(x)
        decoded_y = self.y.decode(y)
        decoded_w = self.w.decode(w)
        decoded_h = self.h.decode(h)

        return decoded_classes, decoded_x, decoded_y, decoded_w, decoded_h
    

    def encode_class(self, x):
        return self.classes.encode(x)
    
    def encode_pos(self, x):
        return self.binning.encode(x)


if __name__ == "__main__":
    classes = [
        "Person",
        "Dog",
        "Table",
        "Television"
        #...
    ]
    classes = Classes(classes=classes)

    binning = Binning(bins=100)

    print(binning.n_tokens)

    encoder = Encoder(classes=classes, binning=binning)

    print(encoder.encode(classes=["Dog", "Table"], x=[0, .3], y=[0, .9], w=[.6, .8], h=[.1, .5]))