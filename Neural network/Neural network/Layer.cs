namespace NeuralNetwork
{
    //Layer of nodes.
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
                    node.value = Activations.SoftMax.Activation(GetLayerNetInputs(),node.GetNetActivationInput());
                } else
                {
                    node.value = Activations.SILU.Activation(node.GetNetActivationInput());
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

