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
    public List<Value> _prev;
    public Consumer<Value> _backward;

    private List<Value> topo;
    private Set<Value> visited;

    public Value(Double data) {
        this.constructor(data, new ArrayList<>());
    }

    public Value(Double data, List<Value> _children) {
        this.constructor(data, _children);
    }

    private void constructor(Double data, List<Value> _children) {
        this.data = data;
        this.grad = 0.0d;
        this._backward = (Value value) -> {};
        this._prev = _children;
    }

    public Value add(Double data) {
        return this.add(new Value(data));
    }

    public Value add(Value other) {

        List<Value> new_children = new ArrayList<>();
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

        List<Value> new_children = new ArrayList<>();
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

    public Value pow(Double other) {

        List<Value> new_children = new ArrayList<>();
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

        List<Value> new_children = new ArrayList<>();
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
        for (Value v : this.topo) {
            v._backward.accept(v);
        }
    }

    public Value neg() { return this.mul(-1.0d); }  // -this

    public Value sub(Double other) { return this.sub(new Value(other)); }  // this - other
    public Value sub(Value other) { return this.add(other.neg()); }  // this - other

    public Value rsub(Double other) { return this.rsub(new Value(other)); }  // other - this
    public Value rsub(Value other) { return other.add(this.neg()); }  // other - this

    public Value div(Double other) { return this.div(new Value(other)); }  // this / other
    public Value div(Value other) { return this.mul(other.pow(-1.0d)); }  // this / other

    public Value rdiv(Double other) { return this.rdiv(new Value(other)); }  // other / this
    public Value rdiv(Value other) { return other.mul(this.pow(-1.0d)); }  // other / this

}
