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
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class demo {

    public static void main(String args[]) {

        System.out.println("\n--- TEST 1 ---");

        Value a = new Value(-4.0);
        Value b = new Value(2.0);
        Value c = a.add(b); 
        Value d = a.mul(b).add(b.pow(3));
        c = c.add(c.add(1.0));
        c = c.add(c.add(1.0).sub(a));
        d = d.add(d.mul(2.0).add(b.add(a).relu()));
        d = d.add(d.mul(3.0).add(b.sub(a).relu()));
        Value e = c.sub(d);
        Value f = e.pow(2.0);
        Value g = f.div(2.0);
        g = g.add(f.rdiv(10.0));
        
        System.out.println(g.data.toString());  // prints 24.7041, the outcome of this forward pass
        
        g.backward();

        System.out.println(a.grad.toString());  // prints 138.8338, i.e. the numerical value of dg/da
        System.out.println(b.grad.toString());  // prints 645.5773, i.e. the numerical value of dg/db


        System.out.println("\n--- TEST 2 ---");

        // initialize a model 
        MLP model = new MLP(2, Arrays.asList(16, 16, 1));  // 2-layer neural network
        System.out.println("number of parameters " + Integer.toString(model.parameters().size()));  // expect 337 parameters

        List<Object> result = loss(model, get_X(), get_y());

        Value total_loss = (Value) result.get(0);
        Double acc = (Double) result.get(1);

        System.out.println(total_loss.data);
        System.out.println(acc.toString());

        for (int k = 0; k < 100; k++) {
            
            // forward
            result = loss(model, get_X(), get_y());

            total_loss = (Value) result.get(0);
            acc = (Double) result.get(1);

            // backward
            model.zero_grad();
            total_loss.backward();

            // update (sgd)
            Double learning_rate = 1.0 - 0.9*k/100;
            for (Value p : model.parameters()) {
                p.data -= learning_rate * p.grad;
            }

            System.out.println("step " + k + " loss " + total_loss.data.toString() + ", accuracy " + acc*100 + "%");

        }

    }

    private static List<List<Double>> get_X() {
        List<List<Double>> X = new ArrayList<>();

        X.add(Arrays.asList(1.12211461, 0.08147717));
        X.add(Arrays.asList(-0.81882941 , 0.05879006));
        X.add(Arrays.asList( 1.61370966 ,-0.1246459 ));
        X.add(Arrays.asList(-0.92300918 , 0.3652289 ));
        X.add(Arrays.asList(0.14385146, 0.04438005));
        X.add(Arrays.asList(0.16447247, 0.11738346));
        X.add(Arrays.asList( 1.33877062 ,-0.23800993));
        X.add(Arrays.asList( 0.87114861 ,-0.42271759));
        X.add(Arrays.asList( 1.83129946 ,-0.14104383));
        X.add(Arrays.asList(0.4875712 , 0.63909283));
        X.add(Arrays.asList(0.03746235, 0.42358809));
        X.add(Arrays.asList(-0.44391685 , 0.89673931));
        X.add(Arrays.asList(-0.81222949 , 0.91209092));
        X.add(Arrays.asList( 1.63552312 ,-0.34999676));
        X.add(Arrays.asList(0.47353904, 0.9573426 ));
        X.add(Arrays.asList(0.75354932, 0.62372714));
        X.add(Arrays.asList( 0.26421282 ,-0.24241983));
        X.add(Arrays.asList( 1.42755726 ,-0.37251036));
        X.add(Arrays.asList(-0.37235606 , 0.95669171));
        X.add(Arrays.asList(-0.96130197 , 0.32609011));
        X.add(Arrays.asList(0.78085847, 0.7974894 ));
        X.add(Arrays.asList( 0.91660903 ,-0.42763844));
        X.add(Arrays.asList( 1.04703809 ,-0.54449247));
        X.add(Arrays.asList(-0.06036305 , 0.11960909));
        X.add(Arrays.asList(0.02918954, 0.306839  ));
        X.add(Arrays.asList(-0.39573225 , 0.89654389));
        X.add(Arrays.asList(-0.10464591 , 1.11788313));
        X.add(Arrays.asList(1.88110004, 0.29920257));
        X.add(Arrays.asList(0.82740878, 0.34497717));
        X.add(Arrays.asList( 1.29777112 ,-0.36654315));
        X.add(Arrays.asList(-0.67689285 , 0.85559957));
        X.add(Arrays.asList(0.52952995, 0.94735594));
        X.add(Arrays.asList(-0.84380229 , 0.60473982));
        X.add(Arrays.asList(0.26598471, 0.88732199));
        X.add(Arrays.asList(0.13740386, 0.39785689));
        X.add(Arrays.asList(-0.91043936 ,-0.09709664));
        X.add(Arrays.asList( 1.33740031 ,-0.36741197));
        X.add(Arrays.asList( 1.02257719 ,-0.39752649));
        X.add(Arrays.asList( 1.02490132 ,-0.5486393 ));
        X.add(Arrays.asList(-0.7508959  , 0.25328772));
        X.add(Arrays.asList(1.20281632, 0.08115382));
        X.add(Arrays.asList(-0.46910212 , 0.78079622));
        X.add(Arrays.asList(0.74083668, 0.45923254));
        X.add(Arrays.asList(0.78690512, 0.76624746));
        X.add(Arrays.asList(-0.13005191 , 1.1193894 ));
        X.add(Arrays.asList( 0.80402306 ,-0.42315447));
        X.add(Arrays.asList( 0.28330367 ,-0.21944071));
        X.add(Arrays.asList(-0.7110554,  0.7116388));
        X.add(Arrays.asList( 0.30262452 ,-0.09338909));
        X.add(Arrays.asList(0.80791463, 0.33653833));
        X.add(Arrays.asList(-0.94162691 , 0.16801858));
        X.add(Arrays.asList( 1.14081485 ,-0.46289718));
        X.add(Arrays.asList(-0.15713752 , 0.93210664));
        X.add(Arrays.asList( 1.7150437  ,-0.18362009));
        X.add(Arrays.asList( 0.37246575 ,-0.12607841));
        X.add(Arrays.asList(-0.74309578 , 0.69895167));
        X.add(Arrays.asList(0.68142489, 0.68563424));
        X.add(Arrays.asList( 0.86861248 ,-0.37278617));
        X.add(Arrays.asList(1.00229575, 0.21687478));
        X.add(Arrays.asList(-1.02219391 , 0.39750821));
        X.add(Arrays.asList(-0.61076258 , 0.83117137));
        X.add(Arrays.asList(-0.77668305 , 0.64360932));
        X.add(Arrays.asList(1.10530001, 0.21916281));
        X.add(Arrays.asList(-0.1789046  , 1.06959774));
        X.add(Arrays.asList(0.40498306, 0.82647834));
        X.add(Arrays.asList(1.814573  , 0.03452538));
        X.add(Arrays.asList(-0.79128563 , 0.20006881));
        X.add(Arrays.asList(1.98173149, 0.46076054));
        X.add(Arrays.asList( 0.73242796 ,-0.39965754));
        X.add(Arrays.asList(2.11141647, 0.18064071));
        X.add(Arrays.asList(2.16205598, 0.49423382));
        X.add(Arrays.asList(0.89624542, 0.46153606));
        X.add(Arrays.asList(0.37349376, 1.04498301));
        X.add(Arrays.asList( 0.58496455 ,-0.32125984));
        X.add(Arrays.asList(0.17632979, 0.19745572));
        X.add(Arrays.asList(0.10911565, 0.48155701));
        X.add(Arrays.asList(0.29753275, 0.99583639));
        X.add(Arrays.asList( 0.09604657 ,-0.04626986));
        X.add(Arrays.asList( 0.47464364 ,-0.10512647));
        X.add(Arrays.asList( 1.11634711 ,-0.41553437));
        X.add(Arrays.asList( 0.55362759 ,-0.42312582));
        X.add(Arrays.asList(0.1895281 , 1.01835655));
        X.add(Arrays.asList( 1.94566914 ,-0.0953034 ));
        X.add(Arrays.asList(-0.08462662 , 1.07262342));
        X.add(Arrays.asList( 1.16858008 ,-0.02848107));
        X.add(Arrays.asList(0.12446847, 1.05725031));
        X.add(Arrays.asList(2.03389924, 0.2847298 ));
        X.add(Arrays.asList(-0.02800054 , 0.17076764));
        X.add(Arrays.asList(0.74093481, 0.41411438));
        X.add(Arrays.asList(0.79394703, 0.5597258 ));
        X.add(Arrays.asList( 0.86874262 ,-0.5301472 ));
        X.add(Arrays.asList( 1.61874435 ,-0.32584529));
        X.add(Arrays.asList( 1.42986564 ,-0.4733421 ));
        X.add(Arrays.asList( 1.97480435 ,-0.17793162));
        X.add(Arrays.asList(1.85356347, 0.34226396));
        X.add(Arrays.asList(1.74912164, 0.02833902));
        X.add(Arrays.asList(-0.68566889 , 0.46535694));
        X.add(Arrays.asList(1.75237435, 0.16452051));
        X.add(Arrays.asList( 0.18078955 ,-0.00029542));
        X.add(Arrays.asList(0.12108297, 1.06555225));

        return X;
    }

    private static List<Double> get_y() {
        return Arrays.asList(0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0).stream()
            .map(v -> (double) v)
            .collect(Collectors.toList());
    }

    public static List<Object> loss(MLP model, List<List<Double>> X, List<Double> y, Integer batch_size) {
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

    public static List<Object> loss(MLP model, List<List<Double>> Xb, List<Double> yb) {

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

        Double acc = ((double) accuracy.stream().mapToInt(Integer::intValue).sum()) / accuracy.size();

        ArrayList<Object> result = new ArrayList<>();
        result.add(total_loss);
        result.add(acc);

        return result;
    }

}
