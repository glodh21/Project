Scale-Consistent Edge Detection for DWT Steganography"

The specific unsolved problem:
Standard edge detectors (Canny, Phase Congruency) produce edges at multiple scales independently. When you apply DWT, edges in subbands (e.g., HL at level 2) don't align with edges in level 1. This forces steganography to either:

· Embed redundantly (reducing payload), or
· Risk destroying edge information (making detection easy).

Your solution: Design an edge detector that outputs nested edge maps—edges at coarse scales are guaranteed subsets of finer scales. This is called scale-space consistency


Output/edges (edge detected by in built cv2.Canny()
Output/newedges (edge detected by the code)

USED FOR COMPARISON ONLY
