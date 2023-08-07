using System.Reflection.Emit;

namespace NeuralNetwork
{
    //Layer of nodes.
    [Serializable]
    public class Layer
    {
        public Node[] nodes;
        public bool outputLayer;
        public bool inputLayer;
        public Layer(Node[] nodes, bool outputLayer, bool inputLayer)
        {
            this.nodes = nodes;
            this.outputLayer = outputLayer;
            this.inputLayer = inputLayer;
        }

        /*
            UPDATE LAYER DATA.
        */
        public void UpdateValues()
        {
            foreach (Node node in nodes)
            {
                if (outputLayer)
                {
                    node.value = Activations.SoftMax.Activation(this,node.GetNetActivationInput());
                } else
                {
                    node.value = Activations.Tanh.Activation(node.GetNetActivationInput());
                }
            }
        }

        public void UpdateLayerGradients(double[] expected)
        {
            for (int j = 0; j < nodes.Length; j++)
            {
                Node node = nodes[j];
                lock (node)
                {
                    node.UpdateGradient(this, (outputLayer) ? expected[j] : 0d);
                    node.AddBiasDerivative();
                    foreach (Connection connection in node.inputConnections)
                    {
                        lock (connection)
                        {
                            connection.AddWeightDerivative();
                        }
                    }
                }
            }
        }

        /*
            SET LAYER DATA.
        */
        public void SetLayerValues(double[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                nodes[i].value = values[i];
            }
        }

        /*
            GET LAYER DATA.
        */
        public double[] GetLayerValues()
        {
            double[] vals = new double[nodes.Length];
            for (int i = 0; i < nodes.Length; i++)
            {
                vals[i] = nodes[i].value;
            }
            return vals;
        }

        public double[] GetLayerNetInputs()
        {
            double[] nets = new double[nodes.Length];
            for(int i = 0;  i < nets.Length; i++)
            {
                nets[i] = nodes[i].GetNetActivationInput();
            }
            return nets;
        }
    }
}

