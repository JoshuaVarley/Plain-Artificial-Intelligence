namespace NeuralNetwork
{
	public readonly struct Activations
	{
        public readonly struct SELU
        {
            public static double Activation(double x)
            {
                if (x<=0d)
                {
                    return 1.758d * (Math.Exp(x) - 1);
                }
                return 1.051d * x;
            }

            public static double Derivative(double x)
            {
                if (x<=0d)
                {
                    return 1.758d * Math.Exp(x);
                }

                return 1.051d;
            }
        }


        public readonly struct RELU
        {
            public static double Activation(double x)
            {
                if (x >= 0d)
                {
                    return x;
                }
                return 0d;
            }
            public static double Derivative(double x)
            {
                if (x >= 0d)
                {
                    return 1d;
                }
                return 0d;
            }
        }

        public readonly struct LeakyRELU
        {
            public static double Activation(double x)
            {
                if (x >= 0d)
                {
                    return x;
                }
                return 0.01d * x;
            }
            public static double Derivative(double x)
            {
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

        public readonly struct SoftPlus
        {
            public static double Activation(double x)
            {
                return Math.Log(1 + Math.Exp(x));
            }

            public static double Derivative(double x)
            {
                return Sigmoid.Activation(x);
            }
        }


        public readonly struct SoftMax
        {
            public static double Activation(Layer layer, double value)
            {
                double[] x = layer.GetLayerNetInputs();
                double expsum = 0d;
                for (int i = 0; i < x.Length; i++)
                {

                    expsum += Math.Exp(x[i]);
                }
                return Math.Exp(value) / expsum;
            }

            public static double Derivative(Layer layer, double val)
            {
                double[] x = layer.GetLayerNetInputs();
                double expsum = 0d;
                for (int i = 0; i < x.Length; i++)
                {
                    expsum += Math.Exp(x[i]);
                }

                double ex = Math.Exp(val);
                return (ex * expsum - ex * ex) / (expsum * expsum);
            }
        }


        public readonly struct Tanh
        {
            public static double Activation(double x)
            {
                return Math.Tanh(x);
            }

            public static double Derivative(double x)
            {
                double val = Math.Tanh(x);
                return 1 - val * val;
            }
        }


    }
}

