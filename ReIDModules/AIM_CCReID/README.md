# AIM-CCReID
An official implement of 《Good is Bad: Causality Inspired Cloth-Debiasing for Cloth-Changing Person Re-Identification》

The testing and evaluation code, with the corresponding weights, are released now.

The full training code will be released soon.

#### Requirements
- Python 3.6
- Pytorch 1.6.0
- yacs
- apex

## Performance of AIM 
<table>
	<tr>
	    <td > </td>
	    <td colspan="4" align="center">RRCC</td>
	    <td colspan="4" align="center">LTCC</td>
	</tr >
	<tr >
      <td>   </td>
	    <td colspan="2" align="center"> Standard</td>
      <td colspan="2" align="center"> Cloth-Changing</td>
	    <td colspan="2" align="center"> Standard</td>
      <td colspan="2" align="center"> Cloth-Changing</td>
	</tr>
	<tr>
	    <td> </td>
      <td>R@1</td>
      <td>mAP</td>
      <td>R@1</td>
      <td>mAP</td>
      <td>R@1</td>
      <td>mAP</td>
      <td>R@1</td>
      <td>mAP</td>
	</tr>
	<tr>
	    <td>Paper</td>
      <td>100.0</td>
      <td>99.9</td>
      <td>57.9</td>
      <td>58.3</td>
      <td>76.3</td>
      <td>41.1</td>
      <td>40.6</td>
      <td>19.1</td>
	</tr>
	<tr>
	    <td>Repo</td>
      <td>100.0</td>
      <td>99.8</td>
      <td>58.2</td>
      <td>58.0</td>
      <td>75.9</td>
      <td>41.7</td>
      <td>40.8</td>
      <td>19.2</td>
	</tr>
</table>
The indicators provided in this repo are broadly the same as those in the paper, and possibly even better (depending on what your focous is)

## Datasets
PRCC is avaiable at [Here](https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view).

LTCC is avaiable at [Here](https://naiq.github.io/LTCC_Perosn_ReID.html).

LaST is avaiable at [Here](https://github.com/shuxjweb/last).

## Testing
The trained models (weights) are avaiable at https://pan.baidu.com/s/1Du1XgoCim6I_bZtNRm3yPw?pwd=v4ly 
code: v4ly 

You will find the testing script for prcc and ltcc at `test_AIM.sh`, then modify the resume path to your own path where you placed the weights file.  

To be noticed, you need to modify the `DATA ROOT` and `OUTPUT` in the `configs/default_img.py` to your own path before testing.


