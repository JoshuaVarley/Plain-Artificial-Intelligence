namespace NeuralNetwork
{
	public readonly struct Cost
	{
		public readonly struct MSE { 
			public static double CostMultipleFunction(double[][] target, double[][] calculated)
			{
				double sum = 0d;
				for(int i = 0; i < target.Length; i++)
				{
					for(int j = 0; j < target[i].Length; j++)
					{
						sum += CostFunctionIteration(target[i][j], calculated[i][j]);
					}
				}
				return sum / (target.Length * target[0].Length);
			}

			public static double CostFunction(double[] target, double[] calculated)
			{
				double sum = 0d;
				for(int i = 0; i < target.Length; i++)
				{
					sum += CostFunctionIteration(target[i], calculated[i]);
				}
				return sum/calculated.Length;
			}

			public static double CostFunctionIteration(double target, double calculated)
			{
				double diff = target-calculated;
				return diff * diff;
			}

			public static double CostFunctionIterationDerivative(double target, double calculated)
			{
				return 2d*(calculated - target);
			}
        }
        public readonly struct CROSS_ENTROPY
        {
            public static double CostMultipleFunction(double[][] target, double[][] calculated)
            {
				double sum = 0d;
				for(int i = 0; i < target.Length; i++)
				{
					for(int j = 0; j < target[i].Length; j++)
					{
						sum += CostFunctionIteration(target[i][j], calculated[i][j]);
					}
				}
				return sum / (target.Length * target[0].Length);
            }

            public static double CostFunction(double[] target, double[] calculated)
            {
				double sum = 0d;
				for (int i = 0; i < target.Length; i++) {
					sum += CostFunctionIteration(target[i], calculated[i]);
				}
				return sum / target.Length;
            }

            public static double CostFunctionIteration(double target, double calculated)
            {
                double x = calculated;
                double y = target;
                double v = (y == 1) ? -Math.Log(x) : -Math.Log(1 - x);
                return double.IsNaN(v) ? 0 : v;
            }

            public static double CostFunctionIterationDerivative(double target, double calculated)
            {
                double x = calculated;
                double y = target;
                if (x == 0 || x == 1)
                {
                    return 0;
                }
				double val = (-x + y) / (x * (x - 1)); ;
				return double.IsNaN(val) ? 0 : val;
            }
        }
    }
}

