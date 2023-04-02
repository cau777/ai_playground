# Chess

![Chess page](https://github.com/cau777/ai_playground/blob/master/docs/screenshots/chess_page.png)

### Training

A chess game is modeled as a tree, where each node is a move and stores the evaluation of the position and some extra information. 
Evaluations are just numbers ranging mainly from -1.0 to 1.0, where high values mean positions that are better for white. The process of playing games is just a loop that chooses a node on the tree and evaluates it further. That means evaluating its children using the AI model and sorting them. In that case, the evaluation of the parent node is no longer relevant, instead, the evaluation of the best (depending on the side that is playing) continuation is used.

The training process is focused on **key positions**. These are positions where the preliminary evaluation of the AI and
the evaluation after exploring that position further differ significantly. You can think like moves that "surprised" the AI.
After found, these key positions are analyzed in depth to try finding concrete result (i.e. checkmate).

### Model

The model is based on the GoogleNet architecture, with convolution operations of various sizes.

![Visual representation of the model](https://lapix.ufsc.br/wp-content/uploads/2018/10/inception-2.png)

After this, there is a MaxPool layer and 3 Dense layers. ReLu is used throughout the model as the activation function.

One problem with this approach is that evaluations range from -1.0 to 1.0, and ReLu layers cancel negative values. The consequence of that was that one side was winning more than the other.
A solution to that problem was to add a layer that takes 2 (usually positive) inputs (A and B) and outputs A - B.

### Caching

Any position in chess shares many similarities with the position that leads to it, for example, a pawn push only changes two squares. Also, convolution operations only consider a small part of the board on each iteration.
Based on these two ideas, we can notice that most of the work of the convolution layers is repeated, and can, therefore, be cached to save a significant amount of resources. Each time the convolution kernel is
applied, the method checks whether all inputs are equal to the cache. If so, it uses the cache. If not, it computes the output in the CPU. However, keeping cores for all the nodes is expensive. So, a method is responsible for records for the nodes that probably won't be further analyzed. Dense layers are never cached.

### Outcome

This project is still under development and has several limitations, but early results are promising.