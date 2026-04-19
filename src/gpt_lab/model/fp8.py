class DisableFP8:
    def __init__(self, model):
        self.model = model
        self.fp8_locations = None
        self._active = False

    def _find_modules(self):
        locations = []
        for name, module in self.model.named_modules():
            if 'Float8' in type(module).__name__:
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                    attr_name = name
                locations.append((parent, attr_name, module))
        return locations

    def __enter__(self, *args, **kwargs):
        if self._active:
            raise RuntimeError("DisableFP8 is not reentrant")

        if self.fp8_locations is None:
            self.fp8_locations = self._find_modules()

        if not self.fp8_locations:
            self._active = True
            return self.model

        for parent, attr_name, fp8_module in self.fp8_locations:
            linear = nn.Linear(
                fp8_module.in_features,
                fp8_module.out_features,
                bias=fp8_module.bias is not None,
                device="meta",
                dtype=fp8_module.weight.dtype,
            )
            linear.weight = fp8_module.weight
            if fp8_module.bias is not None:
                linear.bias = fp8_module.bias

            setattr(parent, attr_name, linear)

        self._active = True
        return self.model

    def __exit__(self, exc_type, exc, tb):
        if not self._active:
            return

        for parent, attr_name, fp8_module in self.fp8_locations:
            setattr(parent, attr_name, fp8_module)

        self._active = False
