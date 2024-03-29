��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02unknown8۸

�
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
: *
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
: *
dtype0
�
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_41/kernel
}
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_41/bias
m
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes
:@*
dtype0
�
gradmaps/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_namegradmaps/kernel
|
#gradmaps/kernel/Read/ReadVariableOpReadVariableOpgradmaps/kernel*'
_output_shapes
:@�*
dtype0
s
gradmaps/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namegradmaps/bias
l
!gradmaps/bias/Read/ReadVariableOpReadVariableOpgradmaps/bias*
_output_shapes	
:�*
dtype0
�
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_42/kernel

$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_42/bias
n
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes	
:�*
dtype0
�
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_43/kernel

$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_43/bias
n
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes	
:�*
dtype0
}
Embedding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*!
shared_nameEmbedding/kernel
v
$Embedding/kernel/Read/ReadVariableOpReadVariableOpEmbedding/kernel*
_output_shapes
:	�@*
dtype0
t
Embedding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameEmbedding/bias
m
"Embedding/bias/Read/ReadVariableOpReadVariableOpEmbedding/bias*
_output_shapes
:@*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:@*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
�2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�1
value�1B�1 B�1
�
c1
c2
p1
c3
p2
c4
c5
f1
	dropout1

d1
dropout2
d2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
R
9	variables
:regularization_losses
;trainable_variables
<	keras_api
R
=	variables
>regularization_losses
?trainable_variables
@	keras_api
h

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
R
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
h

Kkernel
Lbias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
6
Qiter
	Rdecay
Slearning_rate
Tmomentum
f
0
1
2
3
#4
$5
-6
.7
38
49
A10
B11
K12
L13
 
f
0
1
2
3
#4
$5
-6
.7
38
49
A10
B11
K12
L13
�
Ulayer_metrics
	variables
Vlayer_regularization_losses

Wlayers
Xmetrics
regularization_losses
Ynon_trainable_variables
trainable_variables
 
JH
VARIABLE_VALUEconv2d_40/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_40/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
Zlayer_metrics
	variables
[layer_regularization_losses

\layers
]metrics
regularization_losses
^non_trainable_variables
trainable_variables
JH
VARIABLE_VALUEconv2d_41/kernel$c2/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_41/bias"c2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
_layer_metrics
	variables
`layer_regularization_losses

alayers
bmetrics
regularization_losses
cnon_trainable_variables
trainable_variables
 
 
 
�
dlayer_metrics
	variables
elayer_regularization_losses

flayers
gmetrics
 regularization_losses
hnon_trainable_variables
!trainable_variables
IG
VARIABLE_VALUEgradmaps/kernel$c3/kernel/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEgradmaps/bias"c3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
�
ilayer_metrics
%	variables
jlayer_regularization_losses

klayers
lmetrics
&regularization_losses
mnon_trainable_variables
'trainable_variables
 
 
 
�
nlayer_metrics
)	variables
olayer_regularization_losses

players
qmetrics
*regularization_losses
rnon_trainable_variables
+trainable_variables
JH
VARIABLE_VALUEconv2d_42/kernel$c4/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_42/bias"c4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
�
slayer_metrics
/	variables
tlayer_regularization_losses

ulayers
vmetrics
0regularization_losses
wnon_trainable_variables
1trainable_variables
JH
VARIABLE_VALUEconv2d_43/kernel$c5/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_43/bias"c5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
�
xlayer_metrics
5	variables
ylayer_regularization_losses

zlayers
{metrics
6regularization_losses
|non_trainable_variables
7trainable_variables
 
 
 
�
}layer_metrics
9	variables
~layer_regularization_losses

layers
�metrics
:regularization_losses
�non_trainable_variables
;trainable_variables
 
 
 
�
�layer_metrics
=	variables
 �layer_regularization_losses
�layers
�metrics
>regularization_losses
�non_trainable_variables
?trainable_variables
JH
VARIABLE_VALUEEmbedding/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEEmbedding/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1
 

A0
B1
�
�layer_metrics
C	variables
 �layer_regularization_losses
�layers
�metrics
Dregularization_losses
�non_trainable_variables
Etrainable_variables
 
 
 
�
�layer_metrics
G	variables
 �layer_regularization_losses
�layers
�metrics
Hregularization_losses
�non_trainable_variables
Itrainable_variables
IG
VARIABLE_VALUEdense_10/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdense_10/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
�
�layer_metrics
M	variables
 �layer_regularization_losses
�layers
�metrics
Nregularization_losses
�non_trainable_variables
Otrainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
�
serving_default_input_1Placeholder*0
_output_shapes
:����������d*
dtype0*%
shape:����������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasgradmaps/kernelgradmaps/biasconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasEmbedding/kernelEmbedding/biasdense_10/kerneldense_10/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_98521
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp#gradmaps/kernel/Read/ReadVariableOp!gradmaps/bias/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp$Embedding/kernel/Read/ReadVariableOp"Embedding/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_99274
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasgradmaps/kernelgradmaps/biasconv2d_42/kernelconv2d_42/biasconv2d_43/kernelconv2d_43/biasEmbedding/kernelEmbedding/biasdense_10/kerneldense_10/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_99350��	
�
g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_97996

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
E__inference_dropout_16_layer_call_and_return_conditional_losses_99114

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_16_layer_call_fn_99097

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_981772
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_b_frame_cnn_6_layer_call_fn_98587

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�@

unknown_10:@

unknown_11:@

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_981022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_20_layer_call_fn_98986

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_978932
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
D__inference_Embedding_layer_call_and_return_conditional_losses_99138

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_99001

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22@:W S
/
_output_shapes
:���������22@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_97893

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_40_layer_call_and_return_conditional_losses_98961

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������dd 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
-__inference_b_frame_cnn_6_layer_call_fn_98620

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�@

unknown_10:@

unknown_11:@

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_983302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_99165

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�;
�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98102

inputs)
conv2d_40_97947: 
conv2d_40_97949: )
conv2d_41_97964: @
conv2d_41_97966:@)
gradmaps_97987:@�
gradmaps_97989:	�+
conv2d_42_98010:��
conv2d_42_98012:	�+
conv2d_43_98027:��
conv2d_43_98029:	�"
embedding_98065:	�@
embedding_98067:@ 
dense_10_98096:@
dense_10_98098:
identity��!Embedding/StatefulPartitionedCall�!conv2d_40/StatefulPartitionedCall�!conv2d_41/StatefulPartitionedCall�!conv2d_42/StatefulPartitionedCall�!conv2d_43/StatefulPartitionedCall� dense_10/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�"dropout_17/StatefulPartitionedCall� gradmaps/StatefulPartitionedCall�
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_97947conv2d_40_97949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������dd *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_979462#
!conv2d_40/StatefulPartitionedCall�
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0conv2d_41_97964conv2d_41_97966*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������22@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_979632#
!conv2d_41/StatefulPartitionedCall�
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_979732"
 max_pooling2d_20/PartitionedCall�
 gradmaps/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0gradmaps_97987gradmaps_97989*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gradmaps_layer_call_and_return_conditional_losses_979862"
 gradmaps/StatefulPartitionedCall�
 max_pooling2d_21/PartitionedCallPartitionedCall)gradmaps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_979962"
 max_pooling2d_21/PartitionedCall�
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_42_98010conv2d_42_98012*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_980092#
!conv2d_42/StatefulPartitionedCall�
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0conv2d_43_98027conv2d_43_98029*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_980262#
!conv2d_43/StatefulPartitionedCall�
flatten_6/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_980382
flatten_6/PartitionedCall�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_980522$
"dropout_16/StatefulPartitionedCall�
!Embedding/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0embedding_98065embedding_98067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_980642#
!Embedding/StatefulPartitionedCall�
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall*Embedding/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_980822$
"dropout_17/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_10_98096dense_10_98098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_980952"
 dense_10/StatefulPartitionedCall�
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^Embedding/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall!^gradmaps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2F
!Embedding/StatefulPartitionedCall!Embedding/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2D
 gradmaps/StatefulPartitionedCall gradmaps/StatefulPartitionedCall:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�b
�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98725

inputsB
(conv2d_40_conv2d_readvariableop_resource: 7
)conv2d_40_biasadd_readvariableop_resource: B
(conv2d_41_conv2d_readvariableop_resource: @7
)conv2d_41_biasadd_readvariableop_resource:@B
'gradmaps_conv2d_readvariableop_resource:@�7
(gradmaps_biasadd_readvariableop_resource:	�D
(conv2d_42_conv2d_readvariableop_resource:��8
)conv2d_42_biasadd_readvariableop_resource:	�D
(conv2d_43_conv2d_readvariableop_resource:��8
)conv2d_43_biasadd_readvariableop_resource:	�;
(embedding_matmul_readvariableop_resource:	�@7
)embedding_biasadd_readvariableop_resource:@9
'dense_10_matmul_readvariableop_resource:@6
(dense_10_biasadd_readvariableop_resource:
identity�� Embedding/BiasAdd/ReadVariableOp�Embedding/MatMul/ReadVariableOp� conv2d_40/BiasAdd/ReadVariableOp�conv2d_40/Conv2D/ReadVariableOp� conv2d_41/BiasAdd/ReadVariableOp�conv2d_41/Conv2D/ReadVariableOp� conv2d_42/BiasAdd/ReadVariableOp�conv2d_42/Conv2D/ReadVariableOp� conv2d_43/BiasAdd/ReadVariableOp�conv2d_43/Conv2D/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�gradmaps/BiasAdd/ReadVariableOp�gradmaps/Conv2D/ReadVariableOp�
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp�
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2
conv2d_40/Conv2D�
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp�
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/BiasAdd~
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/Relu�
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_41/Conv2D/ReadVariableOp�
conv2d_41/Conv2DConv2Dconv2d_40/Relu:activations:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2
conv2d_41/Conv2D�
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp�
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/BiasAdd~
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_41/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool�
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
gradmaps/Conv2D/ReadVariableOp�
gradmaps/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
gradmaps/Conv2D�
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
gradmaps/BiasAdd/ReadVariableOp�
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
gradmaps/BiasAdd|
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
gradmaps/Relu�
max_pooling2d_21/MaxPoolMaxPoolgradmaps/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool�
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_42/Conv2D/ReadVariableOp�
conv2d_42/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_42/Conv2D�
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp�
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_42/BiasAdd
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_42/Relu�
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_43/Conv2D/ReadVariableOp�
conv2d_43/Conv2DConv2Dconv2d_42/Relu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_43/Conv2D�
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp�
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_43/BiasAdd
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_43/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_43/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_6/Reshapey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Const�
dropout_16/dropout/MulMulflatten_6/Reshape:output:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul~
dropout_16/dropout/ShapeShapeflatten_6/Reshape:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform�
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/y�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_16/dropout/GreaterEqual�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_16/dropout/Cast�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul_1�
Embedding/MatMul/ReadVariableOpReadVariableOp(embedding_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
Embedding/MatMul/ReadVariableOp�
Embedding/MatMulMatMuldropout_16/dropout/Mul_1:z:0'Embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/MatMul�
 Embedding/BiasAdd/ReadVariableOpReadVariableOp)embedding_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 Embedding/BiasAdd/ReadVariableOp�
Embedding/BiasAddBiasAddEmbedding/MatMul:product:0(Embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/BiasAddy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_17/dropout/Const�
dropout_17/dropout/MulMulEmbedding/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul~
dropout_17/dropout/ShapeShapeEmbedding/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape�
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform�
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_17/dropout/GreaterEqual/y�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_17/dropout/GreaterEqual�
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_17/dropout/Cast�
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/BiasAdd|
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_10/Softmaxu
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^Embedding/BiasAdd/ReadVariableOp ^Embedding/MatMul/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2D
 Embedding/BiasAdd/ReadVariableOp Embedding/BiasAdd/ReadVariableOp2B
Embedding/MatMul/ReadVariableOpEmbedding/MatMul/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_98521
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�@

unknown_10:@

unknown_11:@

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_978842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������d
!
_user_specified_name	input_1
�b
�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98869
input_1B
(conv2d_40_conv2d_readvariableop_resource: 7
)conv2d_40_biasadd_readvariableop_resource: B
(conv2d_41_conv2d_readvariableop_resource: @7
)conv2d_41_biasadd_readvariableop_resource:@B
'gradmaps_conv2d_readvariableop_resource:@�7
(gradmaps_biasadd_readvariableop_resource:	�D
(conv2d_42_conv2d_readvariableop_resource:��8
)conv2d_42_biasadd_readvariableop_resource:	�D
(conv2d_43_conv2d_readvariableop_resource:��8
)conv2d_43_biasadd_readvariableop_resource:	�;
(embedding_matmul_readvariableop_resource:	�@7
)embedding_biasadd_readvariableop_resource:@9
'dense_10_matmul_readvariableop_resource:@6
(dense_10_biasadd_readvariableop_resource:
identity�� Embedding/BiasAdd/ReadVariableOp�Embedding/MatMul/ReadVariableOp� conv2d_40/BiasAdd/ReadVariableOp�conv2d_40/Conv2D/ReadVariableOp� conv2d_41/BiasAdd/ReadVariableOp�conv2d_41/Conv2D/ReadVariableOp� conv2d_42/BiasAdd/ReadVariableOp�conv2d_42/Conv2D/ReadVariableOp� conv2d_43/BiasAdd/ReadVariableOp�conv2d_43/Conv2D/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�gradmaps/BiasAdd/ReadVariableOp�gradmaps/Conv2D/ReadVariableOp�
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp�
conv2d_40/Conv2DConv2Dinput_1'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2
conv2d_40/Conv2D�
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp�
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/BiasAdd~
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/Relu�
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_41/Conv2D/ReadVariableOp�
conv2d_41/Conv2DConv2Dconv2d_40/Relu:activations:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2
conv2d_41/Conv2D�
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp�
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/BiasAdd~
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_41/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool�
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
gradmaps/Conv2D/ReadVariableOp�
gradmaps/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
gradmaps/Conv2D�
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
gradmaps/BiasAdd/ReadVariableOp�
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
gradmaps/BiasAdd|
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
gradmaps/Relu�
max_pooling2d_21/MaxPoolMaxPoolgradmaps/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool�
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_42/Conv2D/ReadVariableOp�
conv2d_42/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_42/Conv2D�
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp�
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_42/BiasAdd
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_42/Relu�
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_43/Conv2D/ReadVariableOp�
conv2d_43/Conv2DConv2Dconv2d_42/Relu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_43/Conv2D�
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp�
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_43/BiasAdd
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_43/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_43/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_6/Reshapey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Const�
dropout_16/dropout/MulMulflatten_6/Reshape:output:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul~
dropout_16/dropout/ShapeShapeflatten_6/Reshape:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform�
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/y�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_16/dropout/GreaterEqual�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_16/dropout/Cast�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul_1�
Embedding/MatMul/ReadVariableOpReadVariableOp(embedding_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
Embedding/MatMul/ReadVariableOp�
Embedding/MatMulMatMuldropout_16/dropout/Mul_1:z:0'Embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/MatMul�
 Embedding/BiasAdd/ReadVariableOpReadVariableOp)embedding_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 Embedding/BiasAdd/ReadVariableOp�
Embedding/BiasAddBiasAddEmbedding/MatMul:product:0(Embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/BiasAddy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_17/dropout/Const�
dropout_17/dropout/MulMulEmbedding/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul~
dropout_17/dropout/ShapeShapeEmbedding/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape�
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform�
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_17/dropout/GreaterEqual/y�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_17/dropout/GreaterEqual�
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_17/dropout/Cast�
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/BiasAdd|
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_10/Softmaxu
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^Embedding/BiasAdd/ReadVariableOp ^Embedding/MatMul/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2D
 Embedding/BiasAdd/ReadVariableOp Embedding/BiasAdd/ReadVariableOp2B
Embedding/MatMul/ReadVariableOpEmbedding/MatMul/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:Y U
0
_output_shapes
:����������d
!
_user_specified_name	input_1
�\
�
!__inference__traced_restore_99350
file_prefix;
!assignvariableop_conv2d_40_kernel: /
!assignvariableop_1_conv2d_40_bias: =
#assignvariableop_2_conv2d_41_kernel: @/
!assignvariableop_3_conv2d_41_bias:@=
"assignvariableop_4_gradmaps_kernel:@�/
 assignvariableop_5_gradmaps_bias:	�?
#assignvariableop_6_conv2d_42_kernel:��0
!assignvariableop_7_conv2d_42_bias:	�?
#assignvariableop_8_conv2d_43_kernel:��0
!assignvariableop_9_conv2d_43_bias:	�7
$assignvariableop_10_embedding_kernel:	�@0
"assignvariableop_11_embedding_bias:@5
#assignvariableop_12_dense_10_kernel:@/
!assignvariableop_13_dense_10_bias:&
assignvariableop_14_sgd_iter:	 '
assignvariableop_15_sgd_decay: /
%assignvariableop_16_sgd_learning_rate: *
 assignvariableop_17_sgd_momentum: #
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: 
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB$c4/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c4/bias/.ATTRIBUTES/VARIABLE_VALUEB$c5/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c5/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_40_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_40_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_41_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_41_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_gradmaps_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_gradmaps_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_42_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_42_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_43_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_43_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_embedding_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_embedding_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_sgd_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_sgd_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_sgd_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp assignvariableop_17_sgd_momentumIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22f
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_23�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_99092

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_20_layer_call_fn_98991

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_979732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22@:W S
/
_output_shapes
:���������22@
 
_user_specified_nameinputs
�
F
*__inference_dropout_17_layer_call_fn_99143

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_981512
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_43_layer_call_and_return_conditional_losses_98026

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
E
)__inference_flatten_6_layer_call_fn_99086

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_980382
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_99036

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_41_layer_call_fn_98970

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������22@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_979632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������22@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������dd : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������dd 
 
_user_specified_nameinputs
�b
�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98797

inputsB
(conv2d_40_conv2d_readvariableop_resource: 7
)conv2d_40_biasadd_readvariableop_resource: B
(conv2d_41_conv2d_readvariableop_resource: @7
)conv2d_41_biasadd_readvariableop_resource:@B
'gradmaps_conv2d_readvariableop_resource:@�7
(gradmaps_biasadd_readvariableop_resource:	�D
(conv2d_42_conv2d_readvariableop_resource:��8
)conv2d_42_biasadd_readvariableop_resource:	�D
(conv2d_43_conv2d_readvariableop_resource:��8
)conv2d_43_biasadd_readvariableop_resource:	�;
(embedding_matmul_readvariableop_resource:	�@7
)embedding_biasadd_readvariableop_resource:@9
'dense_10_matmul_readvariableop_resource:@6
(dense_10_biasadd_readvariableop_resource:
identity�� Embedding/BiasAdd/ReadVariableOp�Embedding/MatMul/ReadVariableOp� conv2d_40/BiasAdd/ReadVariableOp�conv2d_40/Conv2D/ReadVariableOp� conv2d_41/BiasAdd/ReadVariableOp�conv2d_41/Conv2D/ReadVariableOp� conv2d_42/BiasAdd/ReadVariableOp�conv2d_42/Conv2D/ReadVariableOp� conv2d_43/BiasAdd/ReadVariableOp�conv2d_43/Conv2D/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�gradmaps/BiasAdd/ReadVariableOp�gradmaps/Conv2D/ReadVariableOp�
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp�
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2
conv2d_40/Conv2D�
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp�
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/BiasAdd~
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/Relu�
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_41/Conv2D/ReadVariableOp�
conv2d_41/Conv2DConv2Dconv2d_40/Relu:activations:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2
conv2d_41/Conv2D�
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp�
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/BiasAdd~
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_41/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool�
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
gradmaps/Conv2D/ReadVariableOp�
gradmaps/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
gradmaps/Conv2D�
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
gradmaps/BiasAdd/ReadVariableOp�
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
gradmaps/BiasAdd|
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
gradmaps/Relu�
max_pooling2d_21/MaxPoolMaxPoolgradmaps/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool�
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_42/Conv2D/ReadVariableOp�
conv2d_42/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_42/Conv2D�
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp�
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_42/BiasAdd
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_42/Relu�
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_43/Conv2D/ReadVariableOp�
conv2d_43/Conv2DConv2Dconv2d_42/Relu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_43/Conv2D�
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp�
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_43/BiasAdd
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_43/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_43/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_6/Reshapey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Const�
dropout_16/dropout/MulMulflatten_6/Reshape:output:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul~
dropout_16/dropout/ShapeShapeflatten_6/Reshape:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform�
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/y�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_16/dropout/GreaterEqual�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_16/dropout/Cast�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul_1�
Embedding/MatMul/ReadVariableOpReadVariableOp(embedding_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
Embedding/MatMul/ReadVariableOp�
Embedding/MatMulMatMuldropout_16/dropout/Mul_1:z:0'Embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/MatMul�
 Embedding/BiasAdd/ReadVariableOpReadVariableOp)embedding_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 Embedding/BiasAdd/ReadVariableOp�
Embedding/BiasAddBiasAddEmbedding/MatMul:product:0(Embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/BiasAddy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_17/dropout/Const�
dropout_17/dropout/MulMulEmbedding/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul~
dropout_17/dropout/ShapeShapeEmbedding/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape�
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform�
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_17/dropout/GreaterEqual/y�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_17/dropout/GreaterEqual�
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_17/dropout/Cast�
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/BiasAdd|
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_10/Softmaxu
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^Embedding/BiasAdd/ReadVariableOp ^Embedding/MatMul/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2D
 Embedding/BiasAdd/ReadVariableOp Embedding/BiasAdd/ReadVariableOp2B
Embedding/MatMul/ReadVariableOpEmbedding/MatMul/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
C__inference_gradmaps_layer_call_and_return_conditional_losses_97986

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_21_layer_call_fn_99026

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_979152
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_98151

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
*__inference_dropout_16_layer_call_fn_99102

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_980522
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�;
�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98330

inputs)
conv2d_40_98289: 
conv2d_40_98291: )
conv2d_41_98294: @
conv2d_41_98296:@)
gradmaps_98300:@�
gradmaps_98302:	�+
conv2d_42_98306:��
conv2d_42_98308:	�+
conv2d_43_98311:��
conv2d_43_98313:	�"
embedding_98318:	�@
embedding_98320:@ 
dense_10_98324:@
dense_10_98326:
identity��!Embedding/StatefulPartitionedCall�!conv2d_40/StatefulPartitionedCall�!conv2d_41/StatefulPartitionedCall�!conv2d_42/StatefulPartitionedCall�!conv2d_43/StatefulPartitionedCall� dense_10/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�"dropout_17/StatefulPartitionedCall� gradmaps/StatefulPartitionedCall�
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40_98289conv2d_40_98291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������dd *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_979462#
!conv2d_40/StatefulPartitionedCall�
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0conv2d_41_98294conv2d_41_98296*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������22@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_979632#
!conv2d_41/StatefulPartitionedCall�
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_979732"
 max_pooling2d_20/PartitionedCall�
 gradmaps/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0gradmaps_98300gradmaps_98302*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gradmaps_layer_call_and_return_conditional_losses_979862"
 gradmaps/StatefulPartitionedCall�
 max_pooling2d_21/PartitionedCallPartitionedCall)gradmaps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_979962"
 max_pooling2d_21/PartitionedCall�
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_42_98306conv2d_42_98308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_980092#
!conv2d_42/StatefulPartitionedCall�
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0conv2d_43_98311conv2d_43_98313*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_980262#
!conv2d_43/StatefulPartitionedCall�
flatten_6/PartitionedCallPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_980382
flatten_6/PartitionedCall�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_980522$
"dropout_16/StatefulPartitionedCall�
!Embedding/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0embedding_98318embedding_98320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_980642#
!Embedding/StatefulPartitionedCall�
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall*Embedding/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_980822$
"dropout_17/StatefulPartitionedCall�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_10_98324dense_10_98326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_980952"
 dense_10/StatefulPartitionedCall�
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^Embedding/StatefulPartitionedCall"^conv2d_40/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_42/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall!^gradmaps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2F
!Embedding/StatefulPartitionedCall!Embedding/StatefulPartitionedCall2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2D
 gradmaps/StatefulPartitionedCall gradmaps/StatefulPartitionedCall:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�b
�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98941
input_1B
(conv2d_40_conv2d_readvariableop_resource: 7
)conv2d_40_biasadd_readvariableop_resource: B
(conv2d_41_conv2d_readvariableop_resource: @7
)conv2d_41_biasadd_readvariableop_resource:@B
'gradmaps_conv2d_readvariableop_resource:@�7
(gradmaps_biasadd_readvariableop_resource:	�D
(conv2d_42_conv2d_readvariableop_resource:��8
)conv2d_42_biasadd_readvariableop_resource:	�D
(conv2d_43_conv2d_readvariableop_resource:��8
)conv2d_43_biasadd_readvariableop_resource:	�;
(embedding_matmul_readvariableop_resource:	�@7
)embedding_biasadd_readvariableop_resource:@9
'dense_10_matmul_readvariableop_resource:@6
(dense_10_biasadd_readvariableop_resource:
identity�� Embedding/BiasAdd/ReadVariableOp�Embedding/MatMul/ReadVariableOp� conv2d_40/BiasAdd/ReadVariableOp�conv2d_40/Conv2D/ReadVariableOp� conv2d_41/BiasAdd/ReadVariableOp�conv2d_41/Conv2D/ReadVariableOp� conv2d_42/BiasAdd/ReadVariableOp�conv2d_42/Conv2D/ReadVariableOp� conv2d_43/BiasAdd/ReadVariableOp�conv2d_43/Conv2D/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�gradmaps/BiasAdd/ReadVariableOp�gradmaps/Conv2D/ReadVariableOp�
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp�
conv2d_40/Conv2DConv2Dinput_1'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2
conv2d_40/Conv2D�
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp�
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/BiasAdd~
conv2d_40/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
conv2d_40/Relu�
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_41/Conv2D/ReadVariableOp�
conv2d_41/Conv2DConv2Dconv2d_40/Relu:activations:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2
conv2d_41/Conv2D�
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp�
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/BiasAdd~
conv2d_41/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
conv2d_41/Relu�
max_pooling2d_20/MaxPoolMaxPoolconv2d_41/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool�
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
gradmaps/Conv2D/ReadVariableOp�
gradmaps/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
gradmaps/Conv2D�
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
gradmaps/BiasAdd/ReadVariableOp�
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
gradmaps/BiasAdd|
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
gradmaps/Relu�
max_pooling2d_21/MaxPoolMaxPoolgradmaps/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool�
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_42/Conv2D/ReadVariableOp�
conv2d_42/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_42/Conv2D�
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp�
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_42/BiasAdd
conv2d_42/ReluReluconv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_42/Relu�
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_43/Conv2D/ReadVariableOp�
conv2d_43/Conv2DConv2Dconv2d_42/Relu:activations:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_43/Conv2D�
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_43/BiasAdd/ReadVariableOp�
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_43/BiasAdd
conv2d_43/ReluReluconv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_43/Relus
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_6/Const�
flatten_6/ReshapeReshapeconv2d_43/Relu:activations:0flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_6/Reshapey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_16/dropout/Const�
dropout_16/dropout/MulMulflatten_6/Reshape:output:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul~
dropout_16/dropout/ShapeShapeflatten_6/Reshape:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform�
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_16/dropout/GreaterEqual/y�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2!
dropout_16/dropout/GreaterEqual�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout_16/dropout/Cast�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout_16/dropout/Mul_1�
Embedding/MatMul/ReadVariableOpReadVariableOp(embedding_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02!
Embedding/MatMul/ReadVariableOp�
Embedding/MatMulMatMuldropout_16/dropout/Mul_1:z:0'Embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/MatMul�
 Embedding/BiasAdd/ReadVariableOpReadVariableOp)embedding_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 Embedding/BiasAdd/ReadVariableOp�
Embedding/BiasAddBiasAddEmbedding/MatMul:product:0(Embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Embedding/BiasAddy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_17/dropout/Const�
dropout_17/dropout/MulMulEmbedding/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul~
dropout_17/dropout/ShapeShapeEmbedding/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape�
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform�
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_17/dropout/GreaterEqual/y�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_17/dropout/GreaterEqual�
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_17/dropout/Cast�
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul_1�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_10/BiasAdd|
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_10/Softmaxu
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^Embedding/BiasAdd/ReadVariableOp ^Embedding/MatMul/ReadVariableOp!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2D
 Embedding/BiasAdd/ReadVariableOp Embedding/BiasAdd/ReadVariableOp2B
Embedding/MatMul/ReadVariableOpEmbedding/MatMul/ReadVariableOp2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:Y U
0
_output_shapes
:����������d
!
_user_specified_name	input_1
�
�
D__inference_conv2d_41_layer_call_and_return_conditional_losses_97963

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������22@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������dd : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������dd 
 
_user_specified_nameinputs
�
�
(__inference_gradmaps_layer_call_fn_99010

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_gradmaps_layer_call_and_return_conditional_losses_979862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
E__inference_dropout_17_layer_call_and_return_conditional_losses_99160

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
E__inference_dropout_16_layer_call_and_return_conditional_losses_98052

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_40_layer_call_fn_98950

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������dd *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_40_layer_call_and_return_conditional_losses_979462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������dd 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
d
E__inference_dropout_17_layer_call_and_return_conditional_losses_98082

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_41_layer_call_and_return_conditional_losses_98981

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������22@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������dd : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������dd 
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_98996

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_98177

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_42_layer_call_and_return_conditional_losses_99061

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_40_layer_call_and_return_conditional_losses_97946

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������dd 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
C__inference_gradmaps_layer_call_and_return_conditional_losses_99021

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_dense_10_layer_call_and_return_conditional_losses_99185

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_conv2d_43_layer_call_fn_99070

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_980262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_21_layer_call_fn_99031

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_979962
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�}
�
 __inference__wrapped_model_97884
input_1P
6b_frame_cnn_6_conv2d_40_conv2d_readvariableop_resource: E
7b_frame_cnn_6_conv2d_40_biasadd_readvariableop_resource: P
6b_frame_cnn_6_conv2d_41_conv2d_readvariableop_resource: @E
7b_frame_cnn_6_conv2d_41_biasadd_readvariableop_resource:@P
5b_frame_cnn_6_gradmaps_conv2d_readvariableop_resource:@�E
6b_frame_cnn_6_gradmaps_biasadd_readvariableop_resource:	�R
6b_frame_cnn_6_conv2d_42_conv2d_readvariableop_resource:��F
7b_frame_cnn_6_conv2d_42_biasadd_readvariableop_resource:	�R
6b_frame_cnn_6_conv2d_43_conv2d_readvariableop_resource:��F
7b_frame_cnn_6_conv2d_43_biasadd_readvariableop_resource:	�I
6b_frame_cnn_6_embedding_matmul_readvariableop_resource:	�@E
7b_frame_cnn_6_embedding_biasadd_readvariableop_resource:@G
5b_frame_cnn_6_dense_10_matmul_readvariableop_resource:@D
6b_frame_cnn_6_dense_10_biasadd_readvariableop_resource:
identity��.b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOp�-b_frame_cnn_6/Embedding/MatMul/ReadVariableOp�.b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOp�-b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOp�.b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOp�-b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOp�.b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOp�-b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOp�.b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOp�-b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOp�-b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOp�,b_frame_cnn_6/dense_10/MatMul/ReadVariableOp�-b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOp�,b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOp�
-b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOpReadVariableOp6b_frame_cnn_6_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOp�
b_frame_cnn_6/conv2d_40/Conv2DConv2Dinput_15b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd *
paddingSAME*
strides
2 
b_frame_cnn_6/conv2d_40/Conv2D�
.b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp7b_frame_cnn_6_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOp�
b_frame_cnn_6/conv2d_40/BiasAddBiasAdd'b_frame_cnn_6/conv2d_40/Conv2D:output:06b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������dd 2!
b_frame_cnn_6/conv2d_40/BiasAdd�
b_frame_cnn_6/conv2d_40/ReluRelu(b_frame_cnn_6/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:���������dd 2
b_frame_cnn_6/conv2d_40/Relu�
-b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOpReadVariableOp6b_frame_cnn_6_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOp�
b_frame_cnn_6/conv2d_41/Conv2DConv2D*b_frame_cnn_6/conv2d_40/Relu:activations:05b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@*
paddingSAME*
strides
2 
b_frame_cnn_6/conv2d_41/Conv2D�
.b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp7b_frame_cnn_6_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOp�
b_frame_cnn_6/conv2d_41/BiasAddBiasAdd'b_frame_cnn_6/conv2d_41/Conv2D:output:06b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������22@2!
b_frame_cnn_6/conv2d_41/BiasAdd�
b_frame_cnn_6/conv2d_41/ReluRelu(b_frame_cnn_6/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:���������22@2
b_frame_cnn_6/conv2d_41/Relu�
&b_frame_cnn_6/max_pooling2d_20/MaxPoolMaxPool*b_frame_cnn_6/conv2d_41/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2(
&b_frame_cnn_6/max_pooling2d_20/MaxPool�
,b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOpReadVariableOp5b_frame_cnn_6_gradmaps_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02.
,b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOp�
b_frame_cnn_6/gradmaps/Conv2DConv2D/b_frame_cnn_6/max_pooling2d_20/MaxPool:output:04b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
b_frame_cnn_6/gradmaps/Conv2D�
-b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOpReadVariableOp6b_frame_cnn_6_gradmaps_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOp�
b_frame_cnn_6/gradmaps/BiasAddBiasAdd&b_frame_cnn_6/gradmaps/Conv2D:output:05b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2 
b_frame_cnn_6/gradmaps/BiasAdd�
b_frame_cnn_6/gradmaps/ReluRelu'b_frame_cnn_6/gradmaps/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
b_frame_cnn_6/gradmaps/Relu�
&b_frame_cnn_6/max_pooling2d_21/MaxPoolMaxPool)b_frame_cnn_6/gradmaps/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2(
&b_frame_cnn_6/max_pooling2d_21/MaxPool�
-b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOpReadVariableOp6b_frame_cnn_6_conv2d_42_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02/
-b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOp�
b_frame_cnn_6/conv2d_42/Conv2DConv2D/b_frame_cnn_6/max_pooling2d_21/MaxPool:output:05b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
b_frame_cnn_6/conv2d_42/Conv2D�
.b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp7b_frame_cnn_6_conv2d_42_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOp�
b_frame_cnn_6/conv2d_42/BiasAddBiasAdd'b_frame_cnn_6/conv2d_42/Conv2D:output:06b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
b_frame_cnn_6/conv2d_42/BiasAdd�
b_frame_cnn_6/conv2d_42/ReluRelu(b_frame_cnn_6/conv2d_42/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
b_frame_cnn_6/conv2d_42/Relu�
-b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOpReadVariableOp6b_frame_cnn_6_conv2d_43_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02/
-b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOp�
b_frame_cnn_6/conv2d_43/Conv2DConv2D*b_frame_cnn_6/conv2d_42/Relu:activations:05b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2 
b_frame_cnn_6/conv2d_43/Conv2D�
.b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp7b_frame_cnn_6_conv2d_43_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype020
.b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOp�
b_frame_cnn_6/conv2d_43/BiasAddBiasAdd'b_frame_cnn_6/conv2d_43/Conv2D:output:06b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2!
b_frame_cnn_6/conv2d_43/BiasAdd�
b_frame_cnn_6/conv2d_43/ReluRelu(b_frame_cnn_6/conv2d_43/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
b_frame_cnn_6/conv2d_43/Relu�
b_frame_cnn_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
b_frame_cnn_6/flatten_6/Const�
b_frame_cnn_6/flatten_6/ReshapeReshape*b_frame_cnn_6/conv2d_43/Relu:activations:0&b_frame_cnn_6/flatten_6/Const:output:0*
T0*(
_output_shapes
:����������2!
b_frame_cnn_6/flatten_6/Reshape�
&b_frame_cnn_6/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&b_frame_cnn_6/dropout_16/dropout/Const�
$b_frame_cnn_6/dropout_16/dropout/MulMul(b_frame_cnn_6/flatten_6/Reshape:output:0/b_frame_cnn_6/dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:����������2&
$b_frame_cnn_6/dropout_16/dropout/Mul�
&b_frame_cnn_6/dropout_16/dropout/ShapeShape(b_frame_cnn_6/flatten_6/Reshape:output:0*
T0*
_output_shapes
:2(
&b_frame_cnn_6/dropout_16/dropout/Shape�
=b_frame_cnn_6/dropout_16/dropout/random_uniform/RandomUniformRandomUniform/b_frame_cnn_6/dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype02?
=b_frame_cnn_6/dropout_16/dropout/random_uniform/RandomUniform�
/b_frame_cnn_6/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/b_frame_cnn_6/dropout_16/dropout/GreaterEqual/y�
-b_frame_cnn_6/dropout_16/dropout/GreaterEqualGreaterEqualFb_frame_cnn_6/dropout_16/dropout/random_uniform/RandomUniform:output:08b_frame_cnn_6/dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������2/
-b_frame_cnn_6/dropout_16/dropout/GreaterEqual�
%b_frame_cnn_6/dropout_16/dropout/CastCast1b_frame_cnn_6/dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������2'
%b_frame_cnn_6/dropout_16/dropout/Cast�
&b_frame_cnn_6/dropout_16/dropout/Mul_1Mul(b_frame_cnn_6/dropout_16/dropout/Mul:z:0)b_frame_cnn_6/dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:����������2(
&b_frame_cnn_6/dropout_16/dropout/Mul_1�
-b_frame_cnn_6/Embedding/MatMul/ReadVariableOpReadVariableOp6b_frame_cnn_6_embedding_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02/
-b_frame_cnn_6/Embedding/MatMul/ReadVariableOp�
b_frame_cnn_6/Embedding/MatMulMatMul*b_frame_cnn_6/dropout_16/dropout/Mul_1:z:05b_frame_cnn_6/Embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2 
b_frame_cnn_6/Embedding/MatMul�
.b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOpReadVariableOp7b_frame_cnn_6_embedding_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOp�
b_frame_cnn_6/Embedding/BiasAddBiasAdd(b_frame_cnn_6/Embedding/MatMul:product:06b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2!
b_frame_cnn_6/Embedding/BiasAdd�
&b_frame_cnn_6/dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&b_frame_cnn_6/dropout_17/dropout/Const�
$b_frame_cnn_6/dropout_17/dropout/MulMul(b_frame_cnn_6/Embedding/BiasAdd:output:0/b_frame_cnn_6/dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2&
$b_frame_cnn_6/dropout_17/dropout/Mul�
&b_frame_cnn_6/dropout_17/dropout/ShapeShape(b_frame_cnn_6/Embedding/BiasAdd:output:0*
T0*
_output_shapes
:2(
&b_frame_cnn_6/dropout_17/dropout/Shape�
=b_frame_cnn_6/dropout_17/dropout/random_uniform/RandomUniformRandomUniform/b_frame_cnn_6/dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02?
=b_frame_cnn_6/dropout_17/dropout/random_uniform/RandomUniform�
/b_frame_cnn_6/dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/b_frame_cnn_6/dropout_17/dropout/GreaterEqual/y�
-b_frame_cnn_6/dropout_17/dropout/GreaterEqualGreaterEqualFb_frame_cnn_6/dropout_17/dropout/random_uniform/RandomUniform:output:08b_frame_cnn_6/dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2/
-b_frame_cnn_6/dropout_17/dropout/GreaterEqual�
%b_frame_cnn_6/dropout_17/dropout/CastCast1b_frame_cnn_6/dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2'
%b_frame_cnn_6/dropout_17/dropout/Cast�
&b_frame_cnn_6/dropout_17/dropout/Mul_1Mul(b_frame_cnn_6/dropout_17/dropout/Mul:z:0)b_frame_cnn_6/dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2(
&b_frame_cnn_6/dropout_17/dropout/Mul_1�
,b_frame_cnn_6/dense_10/MatMul/ReadVariableOpReadVariableOp5b_frame_cnn_6_dense_10_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,b_frame_cnn_6/dense_10/MatMul/ReadVariableOp�
b_frame_cnn_6/dense_10/MatMulMatMul*b_frame_cnn_6/dropout_17/dropout/Mul_1:z:04b_frame_cnn_6/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
b_frame_cnn_6/dense_10/MatMul�
-b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOpReadVariableOp6b_frame_cnn_6_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOp�
b_frame_cnn_6/dense_10/BiasAddBiasAdd'b_frame_cnn_6/dense_10/MatMul:product:05b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
b_frame_cnn_6/dense_10/BiasAdd�
b_frame_cnn_6/dense_10/SoftmaxSoftmax'b_frame_cnn_6/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������2 
b_frame_cnn_6/dense_10/Softmax�
IdentityIdentity(b_frame_cnn_6/dense_10/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp/^b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOp.^b_frame_cnn_6/Embedding/MatMul/ReadVariableOp/^b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOp.^b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOp/^b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOp.^b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOp/^b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOp.^b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOp/^b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOp.^b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOp.^b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOp-^b_frame_cnn_6/dense_10/MatMul/ReadVariableOp.^b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOp-^b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 2`
.b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOp.b_frame_cnn_6/Embedding/BiasAdd/ReadVariableOp2^
-b_frame_cnn_6/Embedding/MatMul/ReadVariableOp-b_frame_cnn_6/Embedding/MatMul/ReadVariableOp2`
.b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOp.b_frame_cnn_6/conv2d_40/BiasAdd/ReadVariableOp2^
-b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOp-b_frame_cnn_6/conv2d_40/Conv2D/ReadVariableOp2`
.b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOp.b_frame_cnn_6/conv2d_41/BiasAdd/ReadVariableOp2^
-b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOp-b_frame_cnn_6/conv2d_41/Conv2D/ReadVariableOp2`
.b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOp.b_frame_cnn_6/conv2d_42/BiasAdd/ReadVariableOp2^
-b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOp-b_frame_cnn_6/conv2d_42/Conv2D/ReadVariableOp2`
.b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOp.b_frame_cnn_6/conv2d_43/BiasAdd/ReadVariableOp2^
-b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOp-b_frame_cnn_6/conv2d_43/Conv2D/ReadVariableOp2^
-b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOp-b_frame_cnn_6/dense_10/BiasAdd/ReadVariableOp2\
,b_frame_cnn_6/dense_10/MatMul/ReadVariableOp,b_frame_cnn_6/dense_10/MatMul/ReadVariableOp2^
-b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOp-b_frame_cnn_6/gradmaps/BiasAdd/ReadVariableOp2\
,b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOp,b_frame_cnn_6/gradmaps/Conv2D/ReadVariableOp:Y U
0
_output_shapes
:����������d
!
_user_specified_name	input_1
�
�
)__inference_Embedding_layer_call_fn_99128

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_Embedding_layer_call_and_return_conditional_losses_980642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_43_layer_call_and_return_conditional_losses_99081

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_10_layer_call_fn_99174

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_980952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_97973

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������22@:W S
/
_output_shapes
:���������22@
 
_user_specified_nameinputs
�1
�
__inference__traced_save_99274
file_prefix/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop.
*savev2_gradmaps_kernel_read_readvariableop,
(savev2_gradmaps_bias_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop/
+savev2_embedding_kernel_read_readvariableop-
)savev2_embedding_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB$c4/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c4/bias/.ATTRIBUTES/VARIABLE_VALUEB$c5/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c5/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop*savev2_gradmaps_kernel_read_readvariableop(savev2_gradmaps_bias_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop+savev2_embedding_kernel_read_readvariableop)savev2_embedding_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:@�:�:��:�:��:�:	�@:@:@:: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_b_frame_cnn_6_layer_call_fn_98554
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�@

unknown_10:@

unknown_11:@

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_981022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������d
!
_user_specified_name	input_1
�
�
)__inference_conv2d_42_layer_call_fn_99050

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv2d_42_layer_call_and_return_conditional_losses_980092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_98038

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_17_layer_call_fn_99148

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_980822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_Embedding_layer_call_and_return_conditional_losses_98064

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_b_frame_cnn_6_layer_call_fn_98653
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�@

unknown_10:@

unknown_11:@

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_983302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������d: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������d
!
_user_specified_name	input_1
�
g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_99041

inputs
identity�
MaxPoolMaxPoolinputs*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
MaxPoolm
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_97915

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_99119

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:����������2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_10_layer_call_and_return_conditional_losses_98095

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_42_layer_call_and_return_conditional_losses_98009

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input_19
serving_default_input_1:0����������d<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
c1
c2
p1
c3
p2
c4
c5
f1
	dropout1

d1
dropout2
d2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"
_tf_keras_model
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
 regularization_losses
!trainable_variables
"	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
)	variables
*regularization_losses
+trainable_variables
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
9	variables
:regularization_losses
;trainable_variables
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
=	variables
>regularization_losses
?trainable_variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Akernel
Bbias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Kkernel
Lbias
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
I
Qiter
	Rdecay
Slearning_rate
Tmomentum"
	optimizer
�
0
1
2
3
#4
$5
-6
.7
38
49
A10
B11
K12
L13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
#4
$5
-6
.7
38
49
A10
B11
K12
L13"
trackable_list_wrapper
�
Ulayer_metrics
	variables
Vlayer_regularization_losses

Wlayers
Xmetrics
regularization_losses
Ynon_trainable_variables
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
*:( 2conv2d_40/kernel
: 2conv2d_40/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Zlayer_metrics
	variables
[layer_regularization_losses

\layers
]metrics
regularization_losses
^non_trainable_variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_41/kernel
:@2conv2d_41/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
_layer_metrics
	variables
`layer_regularization_losses

alayers
bmetrics
regularization_losses
cnon_trainable_variables
trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
dlayer_metrics
	variables
elayer_regularization_losses

flayers
gmetrics
 regularization_losses
hnon_trainable_variables
!trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@�2gradmaps/kernel
:�2gradmaps/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
ilayer_metrics
%	variables
jlayer_regularization_losses

klayers
lmetrics
&regularization_losses
mnon_trainable_variables
'trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
nlayer_metrics
)	variables
olayer_regularization_losses

players
qmetrics
*regularization_losses
rnon_trainable_variables
+trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_42/kernel
:�2conv2d_42/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
slayer_metrics
/	variables
tlayer_regularization_losses

ulayers
vmetrics
0regularization_losses
wnon_trainable_variables
1trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_43/kernel
:�2conv2d_43/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
�
xlayer_metrics
5	variables
ylayer_regularization_losses

zlayers
{metrics
6regularization_losses
|non_trainable_variables
7trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
}layer_metrics
9	variables
~layer_regularization_losses

layers
�metrics
:regularization_losses
�non_trainable_variables
;trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
=	variables
 �layer_regularization_losses
�layers
�metrics
>regularization_losses
�non_trainable_variables
?trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!	�@2Embedding/kernel
:@2Embedding/bias
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
�
�layer_metrics
C	variables
 �layer_regularization_losses
�layers
�metrics
Dregularization_losses
�non_trainable_variables
Etrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
G	variables
 �layer_regularization_losses
�layers
�metrics
Hregularization_losses
�non_trainable_variables
Itrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_10/kernel
:2dense_10/bias
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
�
�layer_metrics
M	variables
 �layer_regularization_losses
�layers
�metrics
Nregularization_losses
�non_trainable_variables
Otrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
�2�
-__inference_b_frame_cnn_6_layer_call_fn_98554
-__inference_b_frame_cnn_6_layer_call_fn_98587
-__inference_b_frame_cnn_6_layer_call_fn_98620
-__inference_b_frame_cnn_6_layer_call_fn_98653�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
 __inference__wrapped_model_97884input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98725
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98797
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98869
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98941�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_40_layer_call_fn_98950�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_40_layer_call_and_return_conditional_losses_98961�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_41_layer_call_fn_98970�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_41_layer_call_and_return_conditional_losses_98981�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_max_pooling2d_20_layer_call_fn_98986
0__inference_max_pooling2d_20_layer_call_fn_98991�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_98996
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_99001�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_gradmaps_layer_call_fn_99010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_gradmaps_layer_call_and_return_conditional_losses_99021�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_max_pooling2d_21_layer_call_fn_99026
0__inference_max_pooling2d_21_layer_call_fn_99031�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_99036
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_99041�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_42_layer_call_fn_99050�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_42_layer_call_and_return_conditional_losses_99061�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_43_layer_call_fn_99070�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_43_layer_call_and_return_conditional_losses_99081�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_6_layer_call_fn_99086�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_6_layer_call_and_return_conditional_losses_99092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dropout_16_layer_call_fn_99097
*__inference_dropout_16_layer_call_fn_99102�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dropout_16_layer_call_and_return_conditional_losses_99114
E__inference_dropout_16_layer_call_and_return_conditional_losses_99119�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_Embedding_layer_call_fn_99128�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_Embedding_layer_call_and_return_conditional_losses_99138�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dropout_17_layer_call_fn_99143
*__inference_dropout_17_layer_call_fn_99148�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dropout_17_layer_call_and_return_conditional_losses_99160
E__inference_dropout_17_layer_call_and_return_conditional_losses_99165�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dense_10_layer_call_fn_99174�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_10_layer_call_and_return_conditional_losses_99185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_98521input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
D__inference_Embedding_layer_call_and_return_conditional_losses_99138]AB0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� }
)__inference_Embedding_layer_call_fn_99128PAB0�-
&�#
!�
inputs����������
� "����������@�
 __inference__wrapped_model_97884�#$-.34ABKL9�6
/�,
*�'
input_1����������d
� "3�0
.
output_1"�
output_1����������
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98725u#$-.34ABKL<�9
2�/
)�&
inputs����������d
p 
� "%�"
�
0���������
� �
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98797u#$-.34ABKL<�9
2�/
)�&
inputs����������d
p
� "%�"
�
0���������
� �
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98869v#$-.34ABKL=�:
3�0
*�'
input_1����������d
p 
� "%�"
�
0���������
� �
H__inference_b_frame_cnn_6_layer_call_and_return_conditional_losses_98941v#$-.34ABKL=�:
3�0
*�'
input_1����������d
p
� "%�"
�
0���������
� �
-__inference_b_frame_cnn_6_layer_call_fn_98554i#$-.34ABKL=�:
3�0
*�'
input_1����������d
p 
� "�����������
-__inference_b_frame_cnn_6_layer_call_fn_98587h#$-.34ABKL<�9
2�/
)�&
inputs����������d
p 
� "�����������
-__inference_b_frame_cnn_6_layer_call_fn_98620h#$-.34ABKL<�9
2�/
)�&
inputs����������d
p
� "�����������
-__inference_b_frame_cnn_6_layer_call_fn_98653i#$-.34ABKL=�:
3�0
*�'
input_1����������d
p
� "�����������
D__inference_conv2d_40_layer_call_and_return_conditional_losses_98961m8�5
.�+
)�&
inputs����������d
� "-�*
#� 
0���������dd 
� �
)__inference_conv2d_40_layer_call_fn_98950`8�5
.�+
)�&
inputs����������d
� " ����������dd �
D__inference_conv2d_41_layer_call_and_return_conditional_losses_98981l7�4
-�*
(�%
inputs���������dd 
� "-�*
#� 
0���������22@
� �
)__inference_conv2d_41_layer_call_fn_98970_7�4
-�*
(�%
inputs���������dd 
� " ����������22@�
D__inference_conv2d_42_layer_call_and_return_conditional_losses_99061n-.8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_42_layer_call_fn_99050a-.8�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_43_layer_call_and_return_conditional_losses_99081n348�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_43_layer_call_fn_99070a348�5
.�+
)�&
inputs����������
� "!������������
C__inference_dense_10_layer_call_and_return_conditional_losses_99185\KL/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� {
(__inference_dense_10_layer_call_fn_99174OKL/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dropout_16_layer_call_and_return_conditional_losses_99114^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
E__inference_dropout_16_layer_call_and_return_conditional_losses_99119^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� 
*__inference_dropout_16_layer_call_fn_99097Q4�1
*�'
!�
inputs����������
p 
� "�����������
*__inference_dropout_16_layer_call_fn_99102Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_dropout_17_layer_call_and_return_conditional_losses_99160\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
E__inference_dropout_17_layer_call_and_return_conditional_losses_99165\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� }
*__inference_dropout_17_layer_call_fn_99143O3�0
)�&
 �
inputs���������@
p 
� "����������@}
*__inference_dropout_17_layer_call_fn_99148O3�0
)�&
 �
inputs���������@
p
� "����������@�
D__inference_flatten_6_layer_call_and_return_conditional_losses_99092b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
)__inference_flatten_6_layer_call_fn_99086U8�5
.�+
)�&
inputs����������
� "������������
C__inference_gradmaps_layer_call_and_return_conditional_losses_99021m#$7�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
(__inference_gradmaps_layer_call_fn_99010`#$7�4
-�*
(�%
inputs���������@
� "!������������
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_98996�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_99001h7�4
-�*
(�%
inputs���������22@
� "-�*
#� 
0���������@
� �
0__inference_max_pooling2d_20_layer_call_fn_98986�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
0__inference_max_pooling2d_20_layer_call_fn_98991[7�4
-�*
(�%
inputs���������22@
� " ����������@�
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_99036�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_99041j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
0__inference_max_pooling2d_21_layer_call_fn_99026�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
0__inference_max_pooling2d_21_layer_call_fn_99031]8�5
.�+
)�&
inputs����������
� "!������������
#__inference_signature_wrapper_98521�#$-.34ABKLD�A
� 
:�7
5
input_1*�'
input_1����������d"3�0
.
output_1"�
output_1���������