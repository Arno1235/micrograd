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

        // a = Value(-4.0)
        // b = Value(2.0)
        // c = a + b
        // d = a * b + b**3
        // c += c + 1
        // c += 1 + c + (-a)
        // d += d * 2 + (b + a).relu()
        // d += 3 * d + (b - a).relu()
        // e = c - d
        // f = e**2
        // g = f / 2.0
        // g += 10.0 / f
        // print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
        // g.backward()
        // print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
        // print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

        Value a = new Value(-4.0d);
        Value b = new Value(2.0d);
        Value c = a.add(b); 
        Value d = (a.mul(b)).add(b.pow(3));
        c = c.add(c.add(1.0d));
        c = c.add(new Value(1.0d).add(c).add(a.mul(-1.0d)));
        d = d.add(d.mul(2.0d).add((b.add(a))).relu());
        d = d.add(new Value(3.0d).mul(d).add((b.add(a.mul(-1.0d)))).relu());
        Value e = c.add(d.mul(-1.0d));
        Value f = e.pow(2.0d);
        Value g = f.mul(new Value(2.0d).pow(-1.0d));
        g = g.add(new Value(10.0d).mul(f.pow(-1.0d)));
        
        System.out.println(g.data.toString());
        
        g.backward();

        System.out.println(a.grad.toString());
        System.out.println(b.grad.toString());
        System.out.println(c.grad.toString());
        System.out.println(d.grad.toString());
        System.out.println(e.grad.toString());
        System.out.println(f.grad.toString());
        System.out.println(g.grad.toString());

    }
}
