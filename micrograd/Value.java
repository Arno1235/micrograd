package micrograd;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import java.util.HashSet;
import java.util.function.Consumer;

public class Value {
    // stores a single scalar value and its gradient

    public Double data;
    public Double grad;
    public Set<Value> _prev;
    public Consumer<Value> _backward;

    private List<Value> topo;
    private Set<Value> visited;

    public Value(Double data) {
        this.constructor(data, new HashSet<>());
    }

    public Value(Double data, Set<Value> _children) {
        this.constructor(data, _children);
    }

    private void constructor(Double data, Set<Value> _children) {
        this.data = data;
        this.grad = 0.0d;
        this._backward = null;
        this._prev = _children;
    }

    public Value add(Double data) {
        return this.add(new Value(data));
    }

    public Value add(Value other) {

        Set<Value> new_children = new HashSet<>();
        new_children.add(this);
        new_children.add(other);

        Value out = new Value(this.data + other.data, new_children);

        out._backward = (Value value) -> {
            for (Value v : value._prev) {
                v.grad += value.grad;
            }
        };

        return out;
    }

    public Value mul(Double data) {
        return this.mul(new Value(data));
    }

    public Value mul(Value other) {

        Set<Value> new_children = new HashSet<>();
        new_children.add(this);
        new_children.add(other);

        Value out = new Value(this.data * other.data, new_children);

        out._backward = (Value value) -> {
            List<Value> temp_children = new ArrayList<Value>();
            temp_children.addAll(value._prev);

            Value value_0 = temp_children.get(0);
            Value value_1 = temp_children.get(1);

            value_0.grad += value_1.data * value.grad;
            value_1.grad += value_0.data * value.grad;
        };

        return out;
    }

    public Value pow(int other) {
        return this.pow((double) other);
    }

    public Value pow(double other) {

        Set<Value> new_children = new HashSet<>();
        new_children.add(this);

        Value out = new Value(Math.pow(this.data, other), new_children);

        out._backward = (Value value) -> {
            for (Value v : value._prev) {
                v.grad += (other * Math.pow(v.data,(other-1))) * value.grad;
            }
        };

        return out;
    }

    public Value relu() {

        Set<Value> new_children = new HashSet<>();
        new_children.add(this);

        Double new_data = 0.0d;
        if (this.data > 0) new_data = this.data;

        Value out = new Value(new_data, new_children);

        out._backward = (Value value) -> {
            for (Value v : value._prev) {
                if (value.data > 0) v.grad += value.grad;
            }
        };

        return out;
    }

    private void build_topo(Value v) {
        if (!this.visited.contains(v)) {
            this.visited.add(v);

            for (Value child : v._prev) this.build_topo(child);
            
            this.topo.add(v);
        }
    }

    public void backward() {

        this.topo = new ArrayList<>();
        this.visited = new HashSet<>();

        this.build_topo(this);

        this.grad = 1.0d;

        Collections.reverse(topo);
        for (Value v : this.topo) v._backward.accept(v);
    }

}