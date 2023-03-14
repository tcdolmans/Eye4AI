class MBT():
    def __init__(self, paradigm, fusion_layer, fusion_token):
        super().__init__()
    self.mode = paradigm
    self.fusion_layer = fusion_layer
    self.fusion_token = fusion_token

    if paradigm == 1:
        