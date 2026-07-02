// to run:
// javac demo.java && java demo

import micrograd.Module;
import micrograd.MLP;
import micrograd.Layer;
import micrograd.Neuron;
import micrograd.Value;

import java.util.ArrayList;

public class demo {
    public static void main(String args[]) {

        Value a = new Value(-4.0d, "a");
        Value b = new Value(2.0d, "b");
        Value c = a.add(b); 
        Value d = (a.mul(b)).add(b.pow(3));
        c = c.add(c.add(1.0d, "1"));
        c = c.add(new Value(1.0d, "1").add(c).add(a.mul(-1.0d, "-1")));
        d = d.add((d.mul(2.0d, "2")).add((b.add(a)).relu()));
        d = d.add(new Value(3.0d, "3").mul(d).add((b.add(a.mul(-1.0d, "-1"))).relu()));
        Value e = c.add(d.mul(-1.0d, "-1"));
        Value f = e.pow(2.0d);
        Value g = f.mul(new Value(2.0d, "2").pow(-1.0d));
        g = g.add(new Value(10.0d, "10").mul(f.pow(-1.0d)));
        
        System.out.println(g.data.toString());  // prints 24.7041, the outcome of this forward pass
        
        g.backward();

        System.out.println(a.grad.toString());  // prints 138.8338, i.e. the numerical value of dg/da
        System.out.println(b.grad.toString());  // prints 645.5773, i.e. the numerical value of dg/db

    }
}
