## Output Format

Once the program is done a single line will be added to the end of the
output file containing 36 values separated by a `;`. In the following, the
meaning of these values is described:

1. The problem id given as part of the input arguments.
2. The id of the used algorithm.
3. The name of the dataset.
4. The number of examples in the dataset.
5. The subset ratio given as part of the input arguments.
6. The subset seed given as part of the input arguments.
7. The number of dimensions in the dataset.
8. The maximum size of the decision tree given as part of the input arguments.
9. The maximum number of seconds that the program was allowed to run.
10. The runtime of the program in milliseconds.
11. The maximum memory in MiB that was used by the program during its runtime.
12. `true` if a timeout happened. `false` otherwise.
13. `true` if the program could find an optimal tree. `false` otherwise.
14. The size of the optimal tree.
15. The ratio of examples that are correctly classified by the optimal tree.
Should always be `1.0`.
16. The number of search tree nodes checked by the algorithm.
17. The number of times the ImpLB caused the algorithm to return from a
search tree node.
18. The number of times the Subset Constraints caused the algorithm to return
from a search tree node.
19. The number of sets saved in the setTrie.
20. The number of times that a subset of an example set was present in the
setTrie and caused the algorithm to return from a search tree node.
21. The number of vertices in the setTrie.
22. The upper bound for the size of the smallest optimal decision tree
given as part of the input arguments.
23. The sum of all leaf depths in the optimal tree.
24. The minimum depth of any leaf in the optimal tree.
25. The maximum depth of any leaf in the optimal tree.
26. The maximum number of dimensions in which two examples differ.
27. The number of cuts in the dataset.
28. The maximum number of unique values in a dimension.
29. How many times the pairLB was calculated.
30. How many times the pairLB caused the algorithm to return from a
search tree node.
31. How many additional search tree nodes were checked due to the subset caching.
32. The amount of time in milliseconds used to calculate pairLBs that caused
the algorithm to return from a search tree node.
33. The amount of time in milliseconds used to calculate pairLBs that did not
cause the algorithm to return from a search tree node.
34. How many times the greedy heuristic caused the program to skip the
calculation of a pairLB.
35. The amount of time in milliseconds used to calculate greedy heuristic.
36. The calculated decision tree as a string. The format of this string is
explained in the section "Tree Output Format".