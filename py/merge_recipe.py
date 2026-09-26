import random

class LZMergeRecipeRandom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodes/Merge"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        seed = kwargs.get("seed", -1)
        if seed is None or int(seed) < 0:
            return float("nan")
        return int(seed)

    def generate(self, seed=-1):
        rng = random.Random(int(seed)) if seed is not None and int(seed) >= 0 else random
        # Generate 19 random values between 0.1-1.0 in 0.1 increments
        values = [round(rng.uniform(0.1, 1.0) / 0.1) * 0.1 for _ in range(19)]
        # Format to ensure proper decimal representation
        formatted_values = [f"{v:.1f}" for v in values]
        recipe = ",".join(formatted_values)
        return (recipe,)


class LZMergeRecipeManual:
    @classmethod
    def INPUT_TYPES(s):
        step_options = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        inputs = {
            "required": {
                "step": (step_options, {"default": 0.1}),
            },
            "optional": {}
        }
        
        # Define 19 individual FLOAT inputs (IN00-IN08, M00, OUT00-OUT08)
        # IN layers (00-08)
        for i in range(9):
            inputs["optional"][f"IN{i:02d}"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1})
        
        # M layer (00)
        inputs["optional"]["M00"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1})
        
        # OUT layers (00-08)
        for i in range(9):
            inputs["optional"][f"OUT{i:02d}"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1})
            
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodes/Merge"

    def generate(self, step=0.1, **kwargs):
        # Collect all 19 values in order
        values = []
        
        # IN layers (00-08)
        for i in range(9):
            key = f"IN{i:02d}"
            value = kwargs.get(key, 0.5)
            # Round to match step increments
            rounded_value = round(value / step) * step
            values.append(rounded_value)
        
        # M layer (00)
        m_value = kwargs.get("M00", 0.5)
        rounded_m_value = round(m_value / step) * step
        values.append(rounded_m_value)
        
        # OUT layers (00-08)
        for i in range(9):
            key = f"OUT{i:02d}"
            value = kwargs.get(key, 0.5)
            # Round to match step increments
            rounded_value = round(value / step) * step
            values.append(rounded_value)
        
        # Format values to match step precision
        # 0.25刻みは小数2桁が必要(0.25/0.75)。0.5/1.0/0.1は1桁でよい
        if step in (0.25, 0.05, 0.01):
            formatted_values = [f"{v:.2f}" for v in values]
        elif step >= 0.1:
            formatted_values = [f"{v:.1f}" for v in values]
        else:
            formatted_values = [str(v) for v in values]

        recipe = ",".join(formatted_values)
        return (recipe,)


class LZMergeRecipeRandomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "step_size": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recipe",)
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodes/Merge"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        seed = kwargs.get("seed", -1)
        if seed is None or int(seed) < 0:
            return float("nan")
        return (int(seed), kwargs.get("min_value"), kwargs.get("max_value"), kwargs.get("step_size"))

    def generate(self, min_value=0.0, max_value=1.0, step_size=0.1, seed=-1):
        # Validate inputs
        if min_value >= max_value:
            raise ValueError("LZMergeRecipeRandomAdvanced Error: min_value must be smaller than max_value.")
        if step_size <= 0:
            step_size = 0.1

        # Calculate possible values within range (浮動小数誤差で終端が落ちないよう丸める)
        num_steps = int(round((max_value - min_value) / step_size)) + 1
        possible_values = [min_value + i * step_size for i in range(num_steps)]

        rng = random.Random(int(seed)) if seed is not None and int(seed) >= 0 else random
        # Generate 19 random values from possible values
        values = [rng.choice(possible_values) for _ in range(19)]

        # Format values to match step precision
        if step_size in (0.25, 0.05, 0.01) or (step_size < 0.1):
            formatted_values = [f"{v:.2f}" for v in values]
        elif step_size >= 0.1:
            formatted_values = [f"{v:.1f}" for v in values]
        else:
            formatted_values = [str(v) for v in values]
            
        recipe = ",".join(formatted_values)
        return (recipe,)