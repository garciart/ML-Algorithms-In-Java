/*
 * The MIT License
 *
 * Copyright 2019 Rob Garcia at rgarcia@rgprogramming.com.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package mlalgorithmsinjava;

import static java.lang.System.exit;

/**
 *
 * @author Rob Garcia at rgarcia@rgprogramming.com
 */
public class MLAlgorithmsInJava {

    /** USE DOUBLE WRAPPER, NOT PRIMITIVE TYPE! WE NEED TO DO NULL CHECKS! **/
    
    /**
     * x represents the set of independent variables, known as the set of features.
     * In this case, there is only one, a house's square footage.
     */
    public static Double[][] xMatrix = {{0.0, 1.0},
                                        {0.0, 2.0},
                                        {0.0, 3.0},
                                        {0.0, 4.0},
                                        {0.0, 5.0}
    };
    
    /**
     * y represents the dependent variable, known as the output feature.
     * In this case, the actual price of the house based on the square footage ($151 per square foot in 2018).
     */
    public static Double[] yVector = {1.0, 2.0, 3.0, 4.0, 5.0};

    /**
     * theta represents the slope of each feature (x) in the set.
     * In this case, the slope of the house's square footage values.
     */
    public static Double[][] thetaMatrix = {{0.0, 0.0},
                                            {0.0, 0.25},
                                            {0.0, 0.50},
                                            {0.0, 0.75},
                                            {0.0, 1.0}
    };
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Hello, world!");
        System.out.println("\nChecking data...\n");
        // Check that the number of x values equals the number of y values.
        if(xMatrix.length != yVector.length) {
            System.out.println("Check your data: the number of x elements (" + xMatrix.length + ") does not equal the number of y elements (" + yVector.length + ").");
            exit(0);
        }
        // Check that there is a theta value for each value of x (per row). 
        else if(xMatrix[0].length != thetaMatrix[0].length) {
            System.out.println("Check your data: the number of x elements (" + xMatrix.length + ") does not equal the number of theta elements (" + thetaMatrix[0].length + ").");
            exit(0);
        }
        // Check that the number of thetas equals the number of y values.
        else if(thetaMatrix.length != yVector.length) {
            System.out.println("Check your data: the number of theta elements (" + thetaMatrix.length + ") does not equal the number of y elements (" + yVector.length + ").");
            exit(0);
        }
        else {
            System.out.println("Data OK.");
            System.out.println("\nCalculating the hypotheses...\n");
            Double[] thetaVector = new Double[thetaMatrix[0].length];
            Double[] xVector = new Double[xMatrix[0].length];
            Double[][] h = new Double[thetaMatrix.length][xMatrix.length];
            for(int i = 0; i < thetaMatrix.length; i++) {
                for(int j = 0; j < xMatrix.length; j++) {
                    for(int k = 0; k < xMatrix[0].length; k++) {
                        xVector[k] = xMatrix[j][k];
                        thetaVector[k] = thetaMatrix[i][k];
                    }

                    if((h[i][j] = LinearRegression.Hypothesis(xVector, thetaVector)) == null) {
                        exit(0);
                    }
                    else {
                        System.out.printf("h(%.2f) = %.2f + (%.2f * %.2f) = %.2f\n", xVector[1], thetaVector[0], thetaVector[1], xVector[1], h[i][j]);
                    }
                }
                System.out.println();
            }
            System.out.println("\nCalculating the cost...\n");
            Double[] J = new Double[xMatrix.length];
            for(int i = 0; i < thetaMatrix.length; i++) {
                if((J[i] = LinearRegression.CostFunction(h[i], yVector)) == null) {
                        exit(0);
                }
                else {
                    System.out.printf("J(%.2f, %.2f) = %.2f\n", thetaMatrix[i][0], thetaMatrix[i][1], J[i]);
                }
            }
            System.out.println("\nPerforming gradient descent...\n");
            Double thetaJ = 0.0;
            Double alpha = 0.0;
            Double[] minimizedThetaVector = new Double[thetaMatrix[0].length];
            /** LEFT OFF HERE! **/
        }
    }
}
