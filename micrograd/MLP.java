package micrograd;

import java.util.List;
import java.util.ArrayList;

public class MLP extends Module {

    private List<Layer> layers;

    public MLP(Integer nin, List<Integer> nouts) {
        List<Integer> sz = new ArrayList<>();
        sz.add(nin);
        sz.addAll(nouts);

        this.layers = new ArrayList<>();
        for (int i = 0;i<nouts.size(); i++) {
            Boolean nonlin = i != nouts.size()-1;
            this.layers.add(
                new Layer(sz.get(i), sz.get(i+1), nonlin)
            );
        }
    }

    public Value call(List<Value> x) {
        for (Layer layer : this.layers) x = layer.call(x);
        return x.get(0);
    }

    public List<Value> parameters() {
        List<Value> p = new ArrayList<>();
        for (Layer l : this.layers) p.addAll(l.parameters());
        return p;
    }

}
