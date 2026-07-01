package micrograd;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class Neuron extends Module {

    private List<Value> w;
    private Value b;
    private Boolean nonlin;

    public Neuron(Integer nin, Boolean nonlin) {
        this.w = new ArrayList<>();

        Random r = new Random();
        for (int i = 0; i<nin; i++) this.w.add(new Value(-1.0d + (1.0d - (-1.0d)) * r.nextDouble()));

        this.b = new Value(0.0d);
        this.nonlin = nonlin;
    }

    public Value call(List<Value> x) {
        assert x.size() == this.w.size();

        Value act = this.w.get(0).mul(x.get(0));

        for (int i = 1; i<x.size(); i++) {
            act = act.add(this.w.get(i).mul(x.get(i)));
        }

        act = act.add(this.b);

        if (this.nonlin) return act.relu();
        return act;
    }

    public List<Value> parameters() {
        List<Value> p = new ArrayList<>();
        p.addAll(this.w);
        p.add(this.b);

        return p;
    }

}
