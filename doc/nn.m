% XOR input for x1 and x2
input = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
output = [0;1;1;0];
% Initialize the bias
bias_h = [1 1 1];
bias_o = [1]
% Learning coefficient
coeff = 0.1;
% Number of learning iterations
iterations = 100000;
% Calculate weights randomly using seed.
%rand('state',sum(100*clock));
weights_bias_h = [0.1; 0.2; 0.3]
weights_bias_o = [0.5]
weights_neuron_h = [0.1 0.2; 0.3 0.4; 0.5 0.6]
weights_neuron_o = [0.6 0.7 0.8]


for i = 1:100000
   out = zeros(4,1);
   numIn = length (input(:,1));
   for j = 1:1%numIn
      % Hidden layer
      %input
      H1 = bias_h(1,1)*weights_bias_h(1,1) + input(j,1)*weights_neuron_h(1,1) + input(j,2)*weights_neuron_h(1,2);
      x2(1) = 1/(1+exp(-H1));
      H2 = bias_h(1,2)*weights_bias_h(2,1) + input(j,1)*weights_neuron_h(2,1) + input(j,2)*weights_neuron_h(2,2);
      x2(2) = 1/(1+exp(-H2));
      H3 = bias_h(1,3)*weights_bias_h(3,1) + input(j,1)*weights_neuron_h(3,1) + input(j,2)*weights_neuron_h(3,2);
      x2(3) = 1/(1+exp(-H3));
      
      % Output layer
      x3_1 = bias_o(1,1)*weights_bias_o(1,1) + x2(1)*weights_neuron_o(1,1) + x2(2)*weights_neuron_o(1,2) + x2(3)*weights_neuron_o(1,3);
      out(j) = 1/(1+exp(-x3_1));
      
      % Adjust delta values of weights
      % For output layer:
      % delta(wi) = xi*delta,
      % delta = (1-actual output)*(desired output - actual output) 
      delta3_1 = out(j)*(1-out(j))*(output(j)-out(j))%;
      
      % Propagate the delta backwards into hidden layers
      delta2_1 = x2(1)*(1-x2(1))*weights_neuron_o(1,1)*delta3_1;
      delta2_2 = x2(2)*(1-x2(2))*weights_neuron_o(1,2)*delta3_1;
      delta2_3 = x2(3)*(1-x2(3))*weights_neuron_o(1,3)*delta3_1;      
      
      % Add weight changes to original weights 
      % And use the new weights to repeat process.
      % delta weight = coeff*x*delta
      % Bias cases
      weights_bias_h(1,1) = 0*weights_bias_h(1,1) + coeff*bias_h(1,1)*delta2_1;
      weights_bias_h(2,1) = 0*weights_bias_h(2,1) + coeff*bias_h(1,2)*delta2_2;
      weights_bias_h(3,1) = 0*weights_bias_h(3,1) + coeff*bias_h(1,3)*delta2_3;
             
      weights_bias_o(1,1) = 0*weights_bias_o(1,1) + coeff*bias_o(1,1)*delta3_1;
             
      % Neuron cases
      weights_neuron_h(1,1) = 0*weights_neuron_h(1,1) + coeff*input(j,1)*delta2_1;
      weights_neuron_h(2,1) = 0*weights_neuron_h(2,1) + coeff*input(j,1)*delta2_2;
      weights_neuron_h(3,1) = 0*weights_neuron_h(3,1) + coeff*input(j,1)*delta2_3;
      
      weights_neuron_h(1,2) = 0*weights_neuron_h(1,2) + coeff*input(j,2)*delta2_1;
      weights_neuron_h(2,2) = 0*weights_neuron_h(2,2) + coeff*input(j,2)*delta2_2;
      weights_neuron_h(3,2) = 0*weights_neuron_h(3,2) + coeff*input(j,2)*delta2_3;
            
      weights_neuron_o(1,1) = 0*weights_neuron_o(1,1) + coeff*x2(1)*delta3_1;
      weights_neuron_o(1,2) = 0*weights_neuron_o(1,2) + coeff*x2(2)*delta3_1;
      weights_neuron_o(1,3) = 0*weights_neuron_o(1,3) + coeff*x2(3)*delta3_1;
      
      weights_bias_h
      weights_bias_o
      weights_neuron_h
      weights_neuron_o
      
   end
end




