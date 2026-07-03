// to run:
// javac demo.java && java demo

import micrograd.Module;
import micrograd.MLP;
import micrograd.Layer;
import micrograd.Neuron;
import micrograd.Value;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import javafx.util.Pair;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class demo {

    public static void main(String args[]) {

        System.out.println("\n--- TEST 1 ---");

        Value a = new Value(-4.0d);
        Value b = new Value(2.0d);
        Value c = a.add(b); 
        Value d = a.mul(b).add(b.pow(3));
        c = c.add(c.add(1.0d));
        c = c.add(c.add(1.0d).sub(a));
        d = d.add(d.mul(2.0d).add(b.add(a).relu()));
        d = d.add(d.mul(3.0d).add(b.sub(a).relu()));
        Value e = c.sub(d);
        Value f = e.pow(2.0d);
        Value g = f.div(2.0d);
        g = g.add(f.rdiv(10.0d));
        
        System.out.println(g.data.toString());  // prints 24.7041, the outcome of this forward pass
        
        g.backward();

        System.out.println(a.grad.toString());  // prints 138.8338, i.e. the numerical value of dg/da
        System.out.println(b.grad.toString());  // prints 645.5773, i.e. the numerical value of dg/db


        System.out.println("\n--- TEST 2 ---");

        // initialize a model 
        MLP model = new MLP(2, Arrays.asList(16, 16, 1));  // 2-layer neural network
        System.out.println("number of parameters " + Integer.toString(model.parameters().size()));  // expect 337 parameters

        loss(model, get_X(), get_y(), 3);

    }

    private static List<List<Double>> get_X() {
        List<List<Double>> X = new ArrayList<>();

        X.add(Arrays.asList(1.0d, 2.0d));
        X.add(Arrays.asList(3.0d, 4.0d));
        X.add(Arrays.asList(5.0d, 6.0d));

        return X;
    }

    private static List<Double> get_y() {
        return Arrays.asList(1.0d, 0.0d, 1.0d);
    }

    public static Pair<Value, Double> loss(MLP model, List<List<Double>> X, List<Double> y, Integer batch_size) {
        assert y.size() == X.size();
        assert batch_size <= y.size();

        List<Integer> indeces = new ArrayList<>();
        for (int i = 0; i<y.size(); i++) indeces.add(i);

        Collections.shuffle(indeces);
        indeces = indeces.subList(0, batch_size);

        List<List<Double>> Xb = indeces.stream()
            .map(index -> X.get(index))
            .collect(Collectors.toList());
        
        List<Double> yb = indeces.stream()
            .map(index -> y.get(index))
            .collect(Collectors.toList());

        return loss(model, Xb, yb);
    }

    public static Pair<Value, Double> loss(MLP model, List<List<Double>> Xb, List<Double> yb) {

        List<List<Value>> inputs = Xb.stream()
            .map(
                xrow -> xrow.stream()
                    .map(itm -> new Value(itm))
                    .collect(Collectors.toList())
            )
            .collect(Collectors.toList());

        // forward the model to get scores
        List<Value> scores = inputs.stream()
            .map(input -> model.call(input))
            .collect(Collectors.toList());

        // svm "max-margin" loss
        List<Value> losses = new ArrayList<>();
        for (int i = 0; i < yb.size(); i++) losses.add( scores.get(i).mul(yb.get(i)).neg().add(1.0d).relu() );
        Value data_loss = losses.stream()
            .reduce(new Value(0.0d), Value::add)
            .mul(1.0d / losses.size());

        // L2 regularization
        Value alpha = new Value(0.0001d);
        Value reg_loss = alpha.mul(
            model.parameters().stream()
                .map(p -> p.mul(p))
                .reduce(new Value(0.0d), Value::add)
        );
        Value total_loss = data_loss.add(reg_loss);

        // also get accuracy
        List<Integer> accuracy = new ArrayList<>();
        for (int i = 0; i < yb.size(); i++) accuracy.add((yb.get(i) > 0) == (scores.get(i).data > 0) ? 1 : 0);

        Double acc = accuracy.stream().sum() / accuracy.size();

        return new Pair(total_loss, acc);
    }

}
