package micrograd;

import java.util.List;
import java.util.ArrayList;

public class Layer extends Module {

    private List<Neuron> neurons;

    public Layer(Integer nin, Integer nout, Boolean nonlin) {
        this.neurons = new ArrayList<>();
        for (int i = 0; i<nout; i++) {
            this.neurons.add(new Neuron(nin, nonlin));
        }
    }

    public List<Value> call(List<Value> x) {
        List<Value> out = new ArrayList<>();

        for (Neuron n : this.neurons) {
            out.add(n.call(x));
        }

        return out;
    }

    public List<Value> parameters() {
        List<Value> p = new ArrayList<>();
        for (Neuron n : this.neurons) p.addAll(n.parameters());
        return p;
    }

}
