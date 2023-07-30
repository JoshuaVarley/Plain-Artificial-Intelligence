namespace NeuralNetwork
{
	public readonly struct Activations
	{

        public readonly struct LeakyRELU
        {
            public static double Activation(double x)
            {
                //LEAKY_RELU
                if (x >= 0d)
                {
                    return x;
                }
                return 0.01d * x;
            }
            public static double Derivative(double x)
            {
                //LEAKY_RELU DERIVATIVE
                if (x >= 0d)
                {
                    return 1d;
                }
                return 0.01d;
            }
        }

        public readonly struct Sigmoid
        {

            public static double Activation(double x)
            {
                return (1.0d) / (1.0d + Math.Exp(-x));
            }

            public static double Derivative(double x)
            {
                double SigX = Activation(x);
                return SigX * (1.0d - SigX);
            }
        }

        public readonly struct SILU
        {
            public static double Activation(double x)
            {
                return x * Sigmoid.Activation(x);
            }

            public static double Derivative(double x)
            {
                double SigX = Sigmoid.Activation(x);
                return SigX*(1 + x * (1-SigX));
            }
        }

        public readonly struct SoftMax
        {
            public static double Activation(double[] x, double value)
            {
                double maxVal = x.Max();
                double logExpSum = 0d;
                for (int i = 0; i < x.Length; i++)
                {
                    logExpSum += Math.Exp(x[i] - maxVal);
                }
                return Math.Exp(value-maxVal-Math.Log(logExpSum));
            }

            public static double Derivative(double[] x, double value)
            {
                double maxVal = x.Max();
                double logExpSum = 0d;
                for(int i = 0; i < x.Length; i++)
                {
                    logExpSum += Math.Exp(value-maxVal);
                }
                logExpSum = maxVal + Math.Log(logExpSum);
                double ex = Math.Exp(value-logExpSum);
                return (ex*Math.Exp(logExpSum)-ex*ex)/(Math.Exp(logExpSum)*Math.Exp(logExpSum));
            }

        }
	}
}

