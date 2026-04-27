# LZ Merge Recipe Nodes

This package provides three nodes for generating merge recipes for model merging in ComfyUI.

## LZMergeRecipeRandom

Generates a random merge recipe with 19 values between 0.1-1.0 in 0.1 increments.

### Inputs
- None

### Output
- `recipe` (STRING): Comma-separated string of 19 random values

## LZMergeRecipeManual

Manually specify all 19 values for a merge recipe with adjustable step size.

### Inputs
- `step` (FLOAT): Step size for all inputs [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] (default: 0.1)
- `IN00` through `IN08` (FLOAT): Input layer weights (default: 0.5)
- `M00` (FLOAT): Middle layer weight (default: 0.5)
- `OUT00` through `OUT08` (FLOAT): Output layer weights (default: 0.5)

### Output
- `recipe` (STRING): Comma-separated string of all 19 values

## LZMergeRecipeRandomAdvanced

Generate a random merge recipe with customizable range and step size.

### Inputs
- `min_value` (FLOAT): Minimum value for generated weights (default: 0.0)
- `max_value` (FLOAT): Maximum value for generated weights (default: 1.0)
- `step_size` (FLOAT): Step increment for generated values (default: 0.1)

### Output
- `recipe` (STRING): Comma-separated string of 19 random values within specified range

## Usage

All nodes output a comma-separated string that can be used with model merging nodes in ComfyUI. The values represent weights for different layers in the model:
- IN00-IN08: Input layers
- M00: Middle layer
- OUT00-OUT08: Output layers

The order of values in the output string is: IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08