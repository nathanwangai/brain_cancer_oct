кЎ	
Ђт
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
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
ѓ
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
2	ѕ
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
Й
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
executor_typestring ѕ
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02unknown8дя
ё
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
: *
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
: *
dtype0
ѓ
gradmaps/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_namegradmaps/kernel
{
#gradmaps/kernel/Read/ReadVariableOpReadVariableOpgradmaps/kernel*&
_output_shapes
:  *
dtype0
r
gradmaps/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namegradmaps/bias
k
!gradmaps/bias/Read/ReadVariableOpReadVariableOpgradmaps/bias*
_output_shapes
: *
dtype0
ё
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_52/kernel
}
$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_52/bias
m
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes
: *
dtype0
ё
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_53/kernel
}
$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*&
_output_shapes
: *
dtype0
t
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_53/bias
m
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes
:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
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
─'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* &
valueш&BЫ& Bв&
┐
c1
p1
c2
p2
c3
c4
f1
dropout1
	d1

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
h

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
6
>iter
	?decay
@learning_rate
Amomentum
F
0
1
2
3
$4
%5
*6
+7
88
99
 
F
0
1
2
3
$4
%5
*6
+7
88
99
Г
Blayer_metrics
	variables
Clayer_regularization_losses

Dlayers
Emetrics
regularization_losses
Fnon_trainable_variables
trainable_variables
 
JH
VARIABLE_VALUEconv2d_51/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_51/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
Glayer_metrics
	variables
Hlayer_regularization_losses

Ilayers
Jmetrics
regularization_losses
Knon_trainable_variables
trainable_variables
 
 
 
Г
Llayer_metrics
	variables
Mlayer_regularization_losses

Nlayers
Ometrics
regularization_losses
Pnon_trainable_variables
trainable_variables
IG
VARIABLE_VALUEgradmaps/kernel$c2/kernel/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEgradmaps/bias"c2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
Qlayer_metrics
	variables
Rlayer_regularization_losses

Slayers
Tmetrics
regularization_losses
Unon_trainable_variables
trainable_variables
 
 
 
Г
Vlayer_metrics
 	variables
Wlayer_regularization_losses

Xlayers
Ymetrics
!regularization_losses
Znon_trainable_variables
"trainable_variables
JH
VARIABLE_VALUEconv2d_52/kernel$c3/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_52/bias"c3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
Г
[layer_metrics
&	variables
\layer_regularization_losses

]layers
^metrics
'regularization_losses
_non_trainable_variables
(trainable_variables
JH
VARIABLE_VALUEconv2d_53/kernel$c4/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d_53/bias"c4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
Г
`layer_metrics
,	variables
alayer_regularization_losses

blayers
cmetrics
-regularization_losses
dnon_trainable_variables
.trainable_variables
 
 
 
Г
elayer_metrics
0	variables
flayer_regularization_losses

glayers
hmetrics
1regularization_losses
inon_trainable_variables
2trainable_variables
 
 
 
Г
jlayer_metrics
4	variables
klayer_regularization_losses

llayers
mmetrics
5regularization_losses
nnon_trainable_variables
6trainable_variables
IG
VARIABLE_VALUEdense_13/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdense_13/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
Г
olayer_metrics
:	variables
player_regularization_losses

qlayers
rmetrics
;regularization_losses
snon_trainable_variables
<trainable_variables
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
?
0
1
2
3
4
5
6
7
	8

t0
u1
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
4
	vtotal
	wcount
x	variables
y	keras_api
D
	ztotal
	{count
|
_fn_kwargs
}	variables
~	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

v0
w1

x	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

z0
{1

}	variables
і
serving_default_input_1Placeholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
№
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_51/kernelconv2d_51/biasgradmaps/kernelgradmaps/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasdense_13/kerneldense_13/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_219901
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ћ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp#gradmaps/kernel/Read/ReadVariableOp!gradmaps/bias/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
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
GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_220455
Д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_51/kernelconv2d_51/biasgradmaps/kernelgradmaps/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasdense_13/kerneldense_13/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
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
GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_220519 Ё
іD
Б
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220151
input_1B
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: A
'gradmaps_conv2d_readvariableop_resource:  6
(gradmaps_biasadd_readvariableop_resource: B
(conv2d_52_conv2d_readvariableop_resource:  7
)conv2d_52_biasadd_readvariableop_resource: B
(conv2d_53_conv2d_readvariableop_resource: 7
)conv2d_53_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:@6
(dense_13_biasadd_readvariableop_resource:
identityѕб conv2d_51/BiasAdd/ReadVariableOpбconv2d_51/Conv2D/ReadVariableOpб conv2d_52/BiasAdd/ReadVariableOpбconv2d_52/Conv2D/ReadVariableOpб conv2d_53/BiasAdd/ReadVariableOpбconv2d_53/Conv2D/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбgradmaps/BiasAdd/ReadVariableOpбgradmaps/Conv2D/ReadVariableOp│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOp┬
conv2d_51/Conv2DConv2Dinput_1'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2
conv2d_51/Conv2Dф
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         22 2
conv2d_51/Relu╩
max_pooling2d_26/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool░
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
gradmaps/Conv2D/ReadVariableOp┘
gradmaps/Conv2DConv2D!max_pooling2d_26/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
gradmaps/Conv2DД
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
gradmaps/BiasAdd/ReadVariableOpг
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
gradmaps/BiasAdd{
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*/
_output_shapes
:          2
gradmaps/Relu╔
max_pooling2d_27/MaxPoolMaxPoolgradmaps/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool│
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_52/Conv2D/ReadVariableOp▄
conv2d_52/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_52/Conv2Dф
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp░
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_52/Relu│
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_53/Conv2D/ReadVariableOpО
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
conv2d_53/Conv2Dф
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp░
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_53/BiasAdds
embedding/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
embedding/ConstЎ
embedding/ReshapeReshapeconv2d_53/BiasAdd:output:0embedding/Const:output:0*
T0*'
_output_shapes
:         @2
embedding/Reshapey
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_21/dropout/Constе
dropout_21/dropout/MulMulembedding/Reshape:output:0!dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul~
dropout_21/dropout/ShapeShapeembedding/Reshape:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeН
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformІ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_21/dropout/GreaterEqual/yЖ
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2!
dropout_21/dropout/GreaterEqualа
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_21/dropout/Castд
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul_1е
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOpц
dense_13/MatMulMatMuldropout_21/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/MatMulД
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpЦ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/BiasAdd|
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_13/Softmaxu
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

IdentityБ
NoOpNoOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1
Ъ
Ъ
*__inference_conv2d_52_layer_call_fn_220290

inputs!
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2195292
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ж
§
D__inference_gradmaps_layer_call_and_return_conditional_losses_220261

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ж
§
D__inference_gradmaps_layer_call_and_return_conditional_losses_219506

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
і
ш
D__inference_dense_13_layer_call_and_return_conditional_losses_220378

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
─L
▒

"__inference__traced_restore_220519
file_prefix;
!assignvariableop_conv2d_51_kernel: /
!assignvariableop_1_conv2d_51_bias: <
"assignvariableop_2_gradmaps_kernel:  .
 assignvariableop_3_gradmaps_bias: =
#assignvariableop_4_conv2d_52_kernel:  /
!assignvariableop_5_conv2d_52_bias: =
#assignvariableop_6_conv2d_53_kernel: /
!assignvariableop_7_conv2d_53_bias:4
"assignvariableop_8_dense_13_kernel:@.
 assignvariableop_9_dense_13_bias:&
assignvariableop_10_sgd_iter:	 '
assignvariableop_11_sgd_decay: /
%assignvariableop_12_sgd_learning_rate: *
 assignvariableop_13_sgd_momentum: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: 
identity_19ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*и
valueГBфB$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB$c4/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names┤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesі
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_51_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_51_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Д
AssignVariableOp_2AssignVariableOp"assignvariableop_2_gradmaps_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_gradmaps_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4е
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_52_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_52_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6е
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_53_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7д
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_53_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_13_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_13_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10ц
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ц
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Г
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13е
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14А
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15А
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Б
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Б
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЖ
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18f
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_19м
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
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
▄
M
1__inference_max_pooling2d_26_layer_call_fn_220226

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2194302
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┐
h
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_219516

inputs
identityњ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
¤

і
$__inference_signature_wrapper_219901
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_2194212
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1
З
ќ
)__inference_dense_13_layer_call_fn_220367

inputs
unknown:@
	unknown_0:
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2195842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ы+
ќ
__inference__traced_save_220455
file_prefix/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop.
*savev2_gradmaps_kernel_read_readvariableop,
(savev2_gradmaps_bias_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЦ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*и
valueГBфB$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB$c4/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c4/bias/.ATTRIBUTES/VARIABLE_VALUEB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names«
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices«
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop*savev2_gradmaps_kernel_read_readvariableop(savev2_gradmaps_bias_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*Ў
_input_shapesЄ
ё: : : :  : :  : : ::@:: : : : : : : : : 2(
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
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: 
і
ш
D__inference_dense_13_layer_call_and_return_conditional_losses_219584

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
■

Њ
.__inference_texture_cnn_5_layer_call_fn_219951

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_2195912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
т
a
E__inference_embedding_layer_call_and_return_conditional_losses_220331

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ю
ъ
)__inference_gradmaps_layer_call_fn_220250

inputs!
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_gradmaps_layer_call_and_return_conditional_losses_2195062
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
г
e
F__inference_dropout_21_layer_call_and_return_conditional_losses_220353

inputs
identityѕc
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
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
іD
Б
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220201
input_1B
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: A
'gradmaps_conv2d_readvariableop_resource:  6
(gradmaps_biasadd_readvariableop_resource: B
(conv2d_52_conv2d_readvariableop_resource:  7
)conv2d_52_biasadd_readvariableop_resource: B
(conv2d_53_conv2d_readvariableop_resource: 7
)conv2d_53_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:@6
(dense_13_biasadd_readvariableop_resource:
identityѕб conv2d_51/BiasAdd/ReadVariableOpбconv2d_51/Conv2D/ReadVariableOpб conv2d_52/BiasAdd/ReadVariableOpбconv2d_52/Conv2D/ReadVariableOpб conv2d_53/BiasAdd/ReadVariableOpбconv2d_53/Conv2D/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбgradmaps/BiasAdd/ReadVariableOpбgradmaps/Conv2D/ReadVariableOp│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOp┬
conv2d_51/Conv2DConv2Dinput_1'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2
conv2d_51/Conv2Dф
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         22 2
conv2d_51/Relu╩
max_pooling2d_26/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool░
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
gradmaps/Conv2D/ReadVariableOp┘
gradmaps/Conv2DConv2D!max_pooling2d_26/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
gradmaps/Conv2DД
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
gradmaps/BiasAdd/ReadVariableOpг
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
gradmaps/BiasAdd{
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*/
_output_shapes
:          2
gradmaps/Relu╔
max_pooling2d_27/MaxPoolMaxPoolgradmaps/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool│
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_52/Conv2D/ReadVariableOp▄
conv2d_52/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_52/Conv2Dф
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp░
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_52/Relu│
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_53/Conv2D/ReadVariableOpО
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
conv2d_53/Conv2Dф
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp░
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_53/BiasAdds
embedding/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
embedding/ConstЎ
embedding/ReshapeReshapeconv2d_53/BiasAdd:output:0embedding/Const:output:0*
T0*'
_output_shapes
:         @2
embedding/Reshapey
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_21/dropout/Constе
dropout_21/dropout/MulMulembedding/Reshape:output:0!dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul~
dropout_21/dropout/ShapeShapeembedding/Reshape:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeН
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformІ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_21/dropout/GreaterEqual/yЖ
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2!
dropout_21/dropout/GreaterEqualа
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_21/dropout/Castд
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul_1е
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOpц
dense_13/MatMulMatMuldropout_21/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/MatMulД
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpЦ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/BiasAdd|
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_13/Softmaxu
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

IdentityБ
NoOpNoOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1
ЄD
б
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220051

inputsB
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: A
'gradmaps_conv2d_readvariableop_resource:  6
(gradmaps_biasadd_readvariableop_resource: B
(conv2d_52_conv2d_readvariableop_resource:  7
)conv2d_52_biasadd_readvariableop_resource: B
(conv2d_53_conv2d_readvariableop_resource: 7
)conv2d_53_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:@6
(dense_13_biasadd_readvariableop_resource:
identityѕб conv2d_51/BiasAdd/ReadVariableOpбconv2d_51/Conv2D/ReadVariableOpб conv2d_52/BiasAdd/ReadVariableOpбconv2d_52/Conv2D/ReadVariableOpб conv2d_53/BiasAdd/ReadVariableOpбconv2d_53/Conv2D/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбgradmaps/BiasAdd/ReadVariableOpбgradmaps/Conv2D/ReadVariableOp│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOp┴
conv2d_51/Conv2DConv2Dinputs'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2
conv2d_51/Conv2Dф
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         22 2
conv2d_51/Relu╩
max_pooling2d_26/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool░
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
gradmaps/Conv2D/ReadVariableOp┘
gradmaps/Conv2DConv2D!max_pooling2d_26/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
gradmaps/Conv2DД
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
gradmaps/BiasAdd/ReadVariableOpг
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
gradmaps/BiasAdd{
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*/
_output_shapes
:          2
gradmaps/Relu╔
max_pooling2d_27/MaxPoolMaxPoolgradmaps/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool│
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_52/Conv2D/ReadVariableOp▄
conv2d_52/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_52/Conv2Dф
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp░
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_52/Relu│
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_53/Conv2D/ReadVariableOpО
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
conv2d_53/Conv2Dф
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp░
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_53/BiasAdds
embedding/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
embedding/ConstЎ
embedding/ReshapeReshapeconv2d_53/BiasAdd:output:0embedding/Const:output:0*
T0*'
_output_shapes
:         @2
embedding/Reshapey
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_21/dropout/Constе
dropout_21/dropout/MulMulembedding/Reshape:output:0!dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul~
dropout_21/dropout/ShapeShapeembedding/Reshape:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeН
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformІ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_21/dropout/GreaterEqual/yЖ
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2!
dropout_21/dropout/GreaterEqualа
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_21/dropout/Castд
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul_1е
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOpц
dense_13/MatMulMatMuldropout_21/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/MatMulД
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpЦ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/BiasAdd|
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_13/Softmaxu
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

IdentityБ
NoOpNoOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Ђ
ћ
.__inference_texture_cnn_5_layer_call_fn_219926
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_2195912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1
Л
F
*__inference_embedding_layer_call_fn_220325

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_2195572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┐
h
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_220241

inputs
identityњ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22 :W S
/
_output_shapes
:         22 
 
_user_specified_nameinputs
№
M
1__inference_max_pooling2d_26_layer_call_fn_220231

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2194932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22 :W S
/
_output_shapes
:         22 
 
_user_specified_nameinputs
ЄD
б
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220101

inputsB
(conv2d_51_conv2d_readvariableop_resource: 7
)conv2d_51_biasadd_readvariableop_resource: A
'gradmaps_conv2d_readvariableop_resource:  6
(gradmaps_biasadd_readvariableop_resource: B
(conv2d_52_conv2d_readvariableop_resource:  7
)conv2d_52_biasadd_readvariableop_resource: B
(conv2d_53_conv2d_readvariableop_resource: 7
)conv2d_53_biasadd_readvariableop_resource:9
'dense_13_matmul_readvariableop_resource:@6
(dense_13_biasadd_readvariableop_resource:
identityѕб conv2d_51/BiasAdd/ReadVariableOpбconv2d_51/Conv2D/ReadVariableOpб conv2d_52/BiasAdd/ReadVariableOpбconv2d_52/Conv2D/ReadVariableOpб conv2d_53/BiasAdd/ReadVariableOpбconv2d_53/Conv2D/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбgradmaps/BiasAdd/ReadVariableOpбgradmaps/Conv2D/ReadVariableOp│
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_51/Conv2D/ReadVariableOp┴
conv2d_51/Conv2DConv2Dinputs'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2
conv2d_51/Conv2Dф
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp░
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         22 2
conv2d_51/Relu╩
max_pooling2d_26/MaxPoolMaxPoolconv2d_51/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPool░
gradmaps/Conv2D/ReadVariableOpReadVariableOp'gradmaps_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
gradmaps/Conv2D/ReadVariableOp┘
gradmaps/Conv2DConv2D!max_pooling2d_26/MaxPool:output:0&gradmaps/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
gradmaps/Conv2DД
gradmaps/BiasAdd/ReadVariableOpReadVariableOp(gradmaps_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
gradmaps/BiasAdd/ReadVariableOpг
gradmaps/BiasAddBiasAddgradmaps/Conv2D:output:0'gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
gradmaps/BiasAdd{
gradmaps/ReluRelugradmaps/BiasAdd:output:0*
T0*/
_output_shapes
:          2
gradmaps/Relu╔
max_pooling2d_27/MaxPoolMaxPoolgradmaps/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_27/MaxPool│
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_52/Conv2D/ReadVariableOp▄
conv2d_52/Conv2DConv2D!max_pooling2d_27/MaxPool:output:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
conv2d_52/Conv2Dф
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp░
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2d_52/BiasAdd~
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:          2
conv2d_52/Relu│
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_53/Conv2D/ReadVariableOpО
conv2d_53/Conv2DConv2Dconv2d_52/Relu:activations:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
conv2d_53/Conv2Dф
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp░
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2d_53/BiasAdds
embedding/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
embedding/ConstЎ
embedding/ReshapeReshapeconv2d_53/BiasAdd:output:0embedding/Const:output:0*
T0*'
_output_shapes
:         @2
embedding/Reshapey
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_21/dropout/Constе
dropout_21/dropout/MulMulembedding/Reshape:output:0!dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul~
dropout_21/dropout/ShapeShapeembedding/Reshape:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeН
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformІ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_21/dropout/GreaterEqual/yЖ
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2!
dropout_21/dropout/GreaterEqualа
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_21/dropout/Castд
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_21/dropout/Mul_1е
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_13/MatMul/ReadVariableOpц
dense_13/MatMulMatMuldropout_21/dropout/Mul_1:z:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/MatMulД
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpЦ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/BiasAdd|
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_13/Softmaxu
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

IdentityБ
NoOpNoOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^gradmaps/BiasAdd/ReadVariableOp^gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
gradmaps/BiasAdd/ReadVariableOpgradmaps/BiasAdd/ReadVariableOp2@
gradmaps/Conv2D/ReadVariableOpgradmaps/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
┐
h
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_219493

inputs
identityњ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         22 :W S
/
_output_shapes
:         22 
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_220236

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_219452

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
■

Њ
.__inference_texture_cnn_5_layer_call_fn_219976

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_2197562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
┐
h
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_220281

inputs
identityњ
MaxPoolMaxPoolinputs*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ъ
Ъ
*__inference_conv2d_53_layer_call_fn_220310

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2195452
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
г
e
F__inference_dropout_21_layer_call_and_return_conditional_losses_219571

inputs
identityѕc
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
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_52_layer_call_and_return_conditional_losses_220301

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├
G
+__inference_dropout_21_layer_call_fn_220336

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_2196322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
т
a
E__inference_embedding_layer_call_and_return_conditional_losses_219557

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Є
■
E__inference_conv2d_53_layer_call_and_return_conditional_losses_220320

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ц
d
+__inference_dropout_21_layer_call_fn_220341

inputs
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_2195712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_220276

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
з
d
F__inference_dropout_21_layer_call_and_return_conditional_losses_219632

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
еV
Њ

!__inference__wrapped_model_219421
input_1P
6texture_cnn_5_conv2d_51_conv2d_readvariableop_resource: E
7texture_cnn_5_conv2d_51_biasadd_readvariableop_resource: O
5texture_cnn_5_gradmaps_conv2d_readvariableop_resource:  D
6texture_cnn_5_gradmaps_biasadd_readvariableop_resource: P
6texture_cnn_5_conv2d_52_conv2d_readvariableop_resource:  E
7texture_cnn_5_conv2d_52_biasadd_readvariableop_resource: P
6texture_cnn_5_conv2d_53_conv2d_readvariableop_resource: E
7texture_cnn_5_conv2d_53_biasadd_readvariableop_resource:G
5texture_cnn_5_dense_13_matmul_readvariableop_resource:@D
6texture_cnn_5_dense_13_biasadd_readvariableop_resource:
identityѕб.texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOpб-texture_cnn_5/conv2d_51/Conv2D/ReadVariableOpб.texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOpб-texture_cnn_5/conv2d_52/Conv2D/ReadVariableOpб.texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOpб-texture_cnn_5/conv2d_53/Conv2D/ReadVariableOpб-texture_cnn_5/dense_13/BiasAdd/ReadVariableOpб,texture_cnn_5/dense_13/MatMul/ReadVariableOpб-texture_cnn_5/gradmaps/BiasAdd/ReadVariableOpб,texture_cnn_5/gradmaps/Conv2D/ReadVariableOpП
-texture_cnn_5/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6texture_cnn_5_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-texture_cnn_5/conv2d_51/Conv2D/ReadVariableOpВ
texture_cnn_5/conv2d_51/Conv2DConv2Dinput_15texture_cnn_5/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2 
texture_cnn_5/conv2d_51/Conv2Dн
.texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7texture_cnn_5_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOpУ
texture_cnn_5/conv2d_51/BiasAddBiasAdd'texture_cnn_5/conv2d_51/Conv2D:output:06texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2!
texture_cnn_5/conv2d_51/BiasAddе
texture_cnn_5/conv2d_51/ReluRelu(texture_cnn_5/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:         22 2
texture_cnn_5/conv2d_51/ReluЗ
&texture_cnn_5/max_pooling2d_26/MaxPoolMaxPool*texture_cnn_5/conv2d_51/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2(
&texture_cnn_5/max_pooling2d_26/MaxPool┌
,texture_cnn_5/gradmaps/Conv2D/ReadVariableOpReadVariableOp5texture_cnn_5_gradmaps_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,texture_cnn_5/gradmaps/Conv2D/ReadVariableOpЉ
texture_cnn_5/gradmaps/Conv2DConv2D/texture_cnn_5/max_pooling2d_26/MaxPool:output:04texture_cnn_5/gradmaps/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
texture_cnn_5/gradmaps/Conv2DЛ
-texture_cnn_5/gradmaps/BiasAdd/ReadVariableOpReadVariableOp6texture_cnn_5_gradmaps_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-texture_cnn_5/gradmaps/BiasAdd/ReadVariableOpС
texture_cnn_5/gradmaps/BiasAddBiasAdd&texture_cnn_5/gradmaps/Conv2D:output:05texture_cnn_5/gradmaps/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2 
texture_cnn_5/gradmaps/BiasAddЦ
texture_cnn_5/gradmaps/ReluRelu'texture_cnn_5/gradmaps/BiasAdd:output:0*
T0*/
_output_shapes
:          2
texture_cnn_5/gradmaps/Reluз
&texture_cnn_5/max_pooling2d_27/MaxPoolMaxPool)texture_cnn_5/gradmaps/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2(
&texture_cnn_5/max_pooling2d_27/MaxPoolП
-texture_cnn_5/conv2d_52/Conv2D/ReadVariableOpReadVariableOp6texture_cnn_5_conv2d_52_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-texture_cnn_5/conv2d_52/Conv2D/ReadVariableOpћ
texture_cnn_5/conv2d_52/Conv2DConv2D/texture_cnn_5/max_pooling2d_27/MaxPool:output:05texture_cnn_5/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2 
texture_cnn_5/conv2d_52/Conv2Dн
.texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp7texture_cnn_5_conv2d_52_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOpУ
texture_cnn_5/conv2d_52/BiasAddBiasAdd'texture_cnn_5/conv2d_52/Conv2D:output:06texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2!
texture_cnn_5/conv2d_52/BiasAddе
texture_cnn_5/conv2d_52/ReluRelu(texture_cnn_5/conv2d_52/BiasAdd:output:0*
T0*/
_output_shapes
:          2
texture_cnn_5/conv2d_52/ReluП
-texture_cnn_5/conv2d_53/Conv2D/ReadVariableOpReadVariableOp6texture_cnn_5_conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-texture_cnn_5/conv2d_53/Conv2D/ReadVariableOpЈ
texture_cnn_5/conv2d_53/Conv2DConv2D*texture_cnn_5/conv2d_52/Relu:activations:05texture_cnn_5/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2 
texture_cnn_5/conv2d_53/Conv2Dн
.texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp7texture_cnn_5_conv2d_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOpУ
texture_cnn_5/conv2d_53/BiasAddBiasAdd'texture_cnn_5/conv2d_53/Conv2D:output:06texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2!
texture_cnn_5/conv2d_53/BiasAddЈ
texture_cnn_5/embedding/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
texture_cnn_5/embedding/ConstЛ
texture_cnn_5/embedding/ReshapeReshape(texture_cnn_5/conv2d_53/BiasAdd:output:0&texture_cnn_5/embedding/Const:output:0*
T0*'
_output_shapes
:         @2!
texture_cnn_5/embedding/ReshapeЋ
&texture_cnn_5/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&texture_cnn_5/dropout_21/dropout/ConstЯ
$texture_cnn_5/dropout_21/dropout/MulMul(texture_cnn_5/embedding/Reshape:output:0/texture_cnn_5/dropout_21/dropout/Const:output:0*
T0*'
_output_shapes
:         @2&
$texture_cnn_5/dropout_21/dropout/Mulе
&texture_cnn_5/dropout_21/dropout/ShapeShape(texture_cnn_5/embedding/Reshape:output:0*
T0*
_output_shapes
:2(
&texture_cnn_5/dropout_21/dropout/Shape 
=texture_cnn_5/dropout_21/dropout/random_uniform/RandomUniformRandomUniform/texture_cnn_5/dropout_21/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02?
=texture_cnn_5/dropout_21/dropout/random_uniform/RandomUniformД
/texture_cnn_5/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/texture_cnn_5/dropout_21/dropout/GreaterEqual/yб
-texture_cnn_5/dropout_21/dropout/GreaterEqualGreaterEqualFtexture_cnn_5/dropout_21/dropout/random_uniform/RandomUniform:output:08texture_cnn_5/dropout_21/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2/
-texture_cnn_5/dropout_21/dropout/GreaterEqual╩
%texture_cnn_5/dropout_21/dropout/CastCast1texture_cnn_5/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2'
%texture_cnn_5/dropout_21/dropout/Castя
&texture_cnn_5/dropout_21/dropout/Mul_1Mul(texture_cnn_5/dropout_21/dropout/Mul:z:0)texture_cnn_5/dropout_21/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2(
&texture_cnn_5/dropout_21/dropout/Mul_1м
,texture_cnn_5/dense_13/MatMul/ReadVariableOpReadVariableOp5texture_cnn_5_dense_13_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,texture_cnn_5/dense_13/MatMul/ReadVariableOp▄
texture_cnn_5/dense_13/MatMulMatMul*texture_cnn_5/dropout_21/dropout/Mul_1:z:04texture_cnn_5/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
texture_cnn_5/dense_13/MatMulЛ
-texture_cnn_5/dense_13/BiasAdd/ReadVariableOpReadVariableOp6texture_cnn_5_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-texture_cnn_5/dense_13/BiasAdd/ReadVariableOpП
texture_cnn_5/dense_13/BiasAddBiasAdd'texture_cnn_5/dense_13/MatMul:product:05texture_cnn_5/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
texture_cnn_5/dense_13/BiasAddд
texture_cnn_5/dense_13/SoftmaxSoftmax'texture_cnn_5/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
texture_cnn_5/dense_13/SoftmaxЃ
IdentityIdentity(texture_cnn_5/dense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

Identity»
NoOpNoOp/^texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOp.^texture_cnn_5/conv2d_51/Conv2D/ReadVariableOp/^texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOp.^texture_cnn_5/conv2d_52/Conv2D/ReadVariableOp/^texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOp.^texture_cnn_5/conv2d_53/Conv2D/ReadVariableOp.^texture_cnn_5/dense_13/BiasAdd/ReadVariableOp-^texture_cnn_5/dense_13/MatMul/ReadVariableOp.^texture_cnn_5/gradmaps/BiasAdd/ReadVariableOp-^texture_cnn_5/gradmaps/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2`
.texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOp.texture_cnn_5/conv2d_51/BiasAdd/ReadVariableOp2^
-texture_cnn_5/conv2d_51/Conv2D/ReadVariableOp-texture_cnn_5/conv2d_51/Conv2D/ReadVariableOp2`
.texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOp.texture_cnn_5/conv2d_52/BiasAdd/ReadVariableOp2^
-texture_cnn_5/conv2d_52/Conv2D/ReadVariableOp-texture_cnn_5/conv2d_52/Conv2D/ReadVariableOp2`
.texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOp.texture_cnn_5/conv2d_53/BiasAdd/ReadVariableOp2^
-texture_cnn_5/conv2d_53/Conv2D/ReadVariableOp-texture_cnn_5/conv2d_53/Conv2D/ReadVariableOp2^
-texture_cnn_5/dense_13/BiasAdd/ReadVariableOp-texture_cnn_5/dense_13/BiasAdd/ReadVariableOp2\
,texture_cnn_5/dense_13/MatMul/ReadVariableOp,texture_cnn_5/dense_13/MatMul/ReadVariableOp2^
-texture_cnn_5/gradmaps/BiasAdd/ReadVariableOp-texture_cnn_5/gradmaps/BiasAdd/ReadVariableOp2\
,texture_cnn_5/gradmaps/Conv2D/ReadVariableOp,texture_cnn_5/gradmaps/Conv2D/ReadVariableOp:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1
№
M
1__inference_max_pooling2d_27_layer_call_fn_220271

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_2195162
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
в
■
E__inference_conv2d_52_layer_call_and_return_conditional_losses_219529

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
з
d
F__inference_dropout_21_layer_call_and_return_conditional_losses_220358

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
в
■
E__inference_conv2d_51_layer_call_and_return_conditional_losses_219483

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         22 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         22 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Ђ
ћ
.__inference_texture_cnn_5_layer_call_fn_220001
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5: 
	unknown_6:
	unknown_7:@
	unknown_8:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *R
fMRK
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_2197562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:         dd
!
_user_specified_name	input_1
в
■
E__inference_conv2d_51_layer_call_and_return_conditional_losses_220221

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         22 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         22 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
г
h
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_219430

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ъ
Ъ
*__inference_conv2d_51_layer_call_fn_220210

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2194832
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         22 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Ё,
»
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_219591

inputs*
conv2d_51_219484: 
conv2d_51_219486: )
gradmaps_219507:  
gradmaps_219509: *
conv2d_52_219530:  
conv2d_52_219532: *
conv2d_53_219546: 
conv2d_53_219548:!
dense_13_219585:@
dense_13_219587:
identityѕб!conv2d_51/StatefulPartitionedCallб!conv2d_52/StatefulPartitionedCallб!conv2d_53/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб"dropout_21/StatefulPartitionedCallб gradmaps/StatefulPartitionedCallц
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_51_219484conv2d_51_219486*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2194832#
!conv2d_51/StatefulPartitionedCallЏ
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2194932"
 max_pooling2d_26/PartitionedCall┬
 gradmaps/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0gradmaps_219507gradmaps_219509*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_gradmaps_layer_call_and_return_conditional_losses_2195062"
 gradmaps/StatefulPartitionedCallџ
 max_pooling2d_27/PartitionedCallPartitionedCall)gradmaps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_2195162"
 max_pooling2d_27/PartitionedCallК
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_27/PartitionedCall:output:0conv2d_52_219530conv2d_52_219532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2195292#
!conv2d_52/StatefulPartitionedCall╚
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_219546conv2d_53_219548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2195452#
!conv2d_53/StatefulPartitionedCall■
embedding/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_2195572
embedding/PartitionedCallЉ
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall"embedding/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_2195712$
"dropout_21/StatefulPartitionedCall╝
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_13_219585dense_13_219587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2195842"
 dense_13/StatefulPartitionedCallё
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityЦ
NoOpNoOp"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall!^gradmaps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2D
 gradmaps/StatefulPartitionedCall gradmaps/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Є
■
E__inference_conv2d_53_layer_call_and_return_conditional_losses_219545

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ё,
»
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_219756

inputs*
conv2d_51_219726: 
conv2d_51_219728: )
gradmaps_219732:  
gradmaps_219734: *
conv2d_52_219738:  
conv2d_52_219740: *
conv2d_53_219743: 
conv2d_53_219745:!
dense_13_219750:@
dense_13_219752:
identityѕб!conv2d_51/StatefulPartitionedCallб!conv2d_52/StatefulPartitionedCallб!conv2d_53/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб"dropout_21/StatefulPartitionedCallб gradmaps/StatefulPartitionedCallц
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_51_219726conv2d_51_219728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         22 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_51_layer_call_and_return_conditional_losses_2194832#
!conv2d_51/StatefulPartitionedCallЏ
 max_pooling2d_26/PartitionedCallPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_2194932"
 max_pooling2d_26/PartitionedCall┬
 gradmaps/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_26/PartitionedCall:output:0gradmaps_219732gradmaps_219734*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_gradmaps_layer_call_and_return_conditional_losses_2195062"
 gradmaps/StatefulPartitionedCallџ
 max_pooling2d_27/PartitionedCallPartitionedCall)gradmaps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_2195162"
 max_pooling2d_27/PartitionedCallК
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_27/PartitionedCall:output:0conv2d_52_219738conv2d_52_219740*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_52_layer_call_and_return_conditional_losses_2195292#
!conv2d_52/StatefulPartitionedCall╚
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0conv2d_53_219743conv2d_53_219745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_conv2d_53_layer_call_and_return_conditional_losses_2195452#
!conv2d_53/StatefulPartitionedCall■
embedding/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_2195572
embedding/PartitionedCallЉ
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall"embedding/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dropout_21_layer_call_and_return_conditional_losses_2195712$
"dropout_21/StatefulPartitionedCall╝
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_13_219750dense_13_219752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_dense_13_layer_call_and_return_conditional_losses_2195842"
 dense_13/StatefulPartitionedCallё
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityЦ
NoOpNoOp"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall!^gradmaps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         dd: : : : : : : : : : 2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2D
 gradmaps/StatefulPartitionedCall gradmaps/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
▄
M
1__inference_max_pooling2d_27_layer_call_fn_220266

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_2194522
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЪ
C
input_18
serving_default_input_1:0         dd<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:»б
▒
c1
p1
c2
p2
c3
c4
f1
dropout1
	d1

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
__call__
ђ_default_save_signature
+Ђ&call_and_return_all_conditional_losses"
_tf_keras_model
й

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
	variables
regularization_losses
trainable_variables
	keras_api
ё__call__
+Ё&call_and_return_all_conditional_losses"
_tf_keras_layer
й

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
є__call__
+Є&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
 	variables
!regularization_losses
"trainable_variables
#	keras_api
ѕ__call__
+Ѕ&call_and_return_all_conditional_losses"
_tf_keras_layer
й

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
і__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
й

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
ї__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
0	variables
1regularization_losses
2trainable_variables
3	keras_api
ј__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_layer
Д
4	variables
5regularization_losses
6trainable_variables
7	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
й

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
њ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
I
>iter
	?decay
@learning_rate
Amomentum"
	optimizer
f
0
1
2
3
$4
%5
*6
+7
88
99"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
$4
%5
*6
+7
88
99"
trackable_list_wrapper
═
Blayer_metrics
	variables
Clayer_regularization_losses

Dlayers
Emetrics
regularization_losses
Fnon_trainable_variables
trainable_variables
__call__
ђ_default_save_signature
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
-
ћserving_default"
signature_map
*:( 2conv2d_51/kernel
: 2conv2d_51/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Glayer_metrics
	variables
Hlayer_regularization_losses

Ilayers
Jmetrics
regularization_losses
Knon_trainable_variables
trainable_variables
ѓ__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Llayer_metrics
	variables
Mlayer_regularization_losses

Nlayers
Ometrics
regularization_losses
Pnon_trainable_variables
trainable_variables
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
):'  2gradmaps/kernel
: 2gradmaps/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
Qlayer_metrics
	variables
Rlayer_regularization_losses

Slayers
Tmetrics
regularization_losses
Unon_trainable_variables
trainable_variables
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Vlayer_metrics
 	variables
Wlayer_regularization_losses

Xlayers
Ymetrics
!regularization_losses
Znon_trainable_variables
"trainable_variables
ѕ__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_52/kernel
: 2conv2d_52/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
░
[layer_metrics
&	variables
\layer_regularization_losses

]layers
^metrics
'regularization_losses
_non_trainable_variables
(trainable_variables
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_53/kernel
:2conv2d_53/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
░
`layer_metrics
,	variables
alayer_regularization_losses

blayers
cmetrics
-regularization_losses
dnon_trainable_variables
.trainable_variables
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
elayer_metrics
0	variables
flayer_regularization_losses

glayers
hmetrics
1regularization_losses
inon_trainable_variables
2trainable_variables
ј__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
jlayer_metrics
4	variables
klayer_regularization_losses

llayers
mmetrics
5regularization_losses
nnon_trainable_variables
6trainable_variables
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_13/kernel
:2dense_13/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
░
olayer_metrics
:	variables
player_regularization_losses

qlayers
rmetrics
;regularization_losses
snon_trainable_variables
<trainable_variables
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
t0
u1"
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
N
	vtotal
	wcount
x	variables
y	keras_api"
_tf_keras_metric
^
	ztotal
	{count
|
_fn_kwargs
}	variables
~	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
v0
w1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
-
}	variables"
_generic_user_object
щ2Ш
.__inference_texture_cnn_5_layer_call_fn_219926
.__inference_texture_cnn_5_layer_call_fn_219951
.__inference_texture_cnn_5_layer_call_fn_219976
.__inference_texture_cnn_5_layer_call_fn_220001│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╠B╔
!__inference__wrapped_model_219421input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
т2Р
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220051
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220101
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220151
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220201│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_51_layer_call_fn_220210б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_51_layer_call_and_return_conditional_losses_220221б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ј2І
1__inference_max_pooling2d_26_layer_call_fn_220226
1__inference_max_pooling2d_26_layer_call_fn_220231б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_220236
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_220241б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_gradmaps_layer_call_fn_220250б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_gradmaps_layer_call_and_return_conditional_losses_220261б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ј2І
1__inference_max_pooling2d_27_layer_call_fn_220266
1__inference_max_pooling2d_27_layer_call_fn_220271б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
─2┴
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_220276
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_220281б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_52_layer_call_fn_220290б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_52_layer_call_and_return_conditional_losses_220301б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_conv2d_53_layer_call_fn_220310б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_conv2d_53_layer_call_and_return_conditional_losses_220320б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_embedding_layer_call_fn_220325б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_embedding_layer_call_and_return_conditional_losses_220331б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_21_layer_call_fn_220336
+__inference_dropout_21_layer_call_fn_220341┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_21_layer_call_and_return_conditional_losses_220353
F__inference_dropout_21_layer_call_and_return_conditional_losses_220358┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_13_layer_call_fn_220367б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_13_layer_call_and_return_conditional_losses_220378б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╦B╚
$__inference_signature_wrapper_219901input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 а
!__inference__wrapped_model_219421{
$%*+898б5
.б+
)і&
input_1         dd
ф "3ф0
.
output_1"і
output_1         х
E__inference_conv2d_51_layer_call_and_return_conditional_losses_220221l7б4
-б*
(і%
inputs         dd
ф "-б*
#і 
0         22 
џ Ї
*__inference_conv2d_51_layer_call_fn_220210_7б4
-б*
(і%
inputs         dd
ф " і         22 х
E__inference_conv2d_52_layer_call_and_return_conditional_losses_220301l$%7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ Ї
*__inference_conv2d_52_layer_call_fn_220290_$%7б4
-б*
(і%
inputs          
ф " і          х
E__inference_conv2d_53_layer_call_and_return_conditional_losses_220320l*+7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         
џ Ї
*__inference_conv2d_53_layer_call_fn_220310_*+7б4
-б*
(і%
inputs          
ф " і         ц
D__inference_dense_13_layer_call_and_return_conditional_losses_220378\89/б,
%б"
 і
inputs         @
ф "%б"
і
0         
џ |
)__inference_dense_13_layer_call_fn_220367O89/б,
%б"
 і
inputs         @
ф "і         д
F__inference_dropout_21_layer_call_and_return_conditional_losses_220353\3б0
)б&
 і
inputs         @
p
ф "%б"
і
0         @
џ д
F__inference_dropout_21_layer_call_and_return_conditional_losses_220358\3б0
)б&
 і
inputs         @
p 
ф "%б"
і
0         @
џ ~
+__inference_dropout_21_layer_call_fn_220336O3б0
)б&
 і
inputs         @
p 
ф "і         @~
+__inference_dropout_21_layer_call_fn_220341O3б0
)б&
 і
inputs         @
p
ф "і         @Е
E__inference_embedding_layer_call_and_return_conditional_losses_220331`7б4
-б*
(і%
inputs         
ф "%б"
і
0         @
џ Ђ
*__inference_embedding_layer_call_fn_220325S7б4
-б*
(і%
inputs         
ф "і         @┤
D__inference_gradmaps_layer_call_and_return_conditional_losses_220261l7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ ї
)__inference_gradmaps_layer_call_fn_220250_7б4
-б*
(і%
inputs          
ф " і          №
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_220236ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ И
L__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_220241h7б4
-б*
(і%
inputs         22 
ф "-б*
#і 
0          
џ К
1__inference_max_pooling2d_26_layer_call_fn_220226ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    љ
1__inference_max_pooling2d_26_layer_call_fn_220231[7б4
-б*
(і%
inputs         22 
ф " і          №
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_220276ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ И
L__inference_max_pooling2d_27_layer_call_and_return_conditional_losses_220281h7б4
-б*
(і%
inputs          
ф "-б*
#і 
0          
џ К
1__inference_max_pooling2d_27_layer_call_fn_220266ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    љ
1__inference_max_pooling2d_27_layer_call_fn_220271[7б4
-б*
(і%
inputs          
ф " і          »
$__inference_signature_wrapper_219901є
$%*+89Cб@
б 
9ф6
4
input_1)і&
input_1         dd"3ф0
.
output_1"і
output_1         й
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220051p
$%*+89;б8
1б.
(і%
inputs         dd
p 
ф "%б"
і
0         
џ й
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220101p
$%*+89;б8
1б.
(і%
inputs         dd
p
ф "%б"
і
0         
џ Й
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220151q
$%*+89<б9
2б/
)і&
input_1         dd
p 
ф "%б"
і
0         
џ Й
I__inference_texture_cnn_5_layer_call_and_return_conditional_losses_220201q
$%*+89<б9
2б/
)і&
input_1         dd
p
ф "%б"
і
0         
џ ќ
.__inference_texture_cnn_5_layer_call_fn_219926d
$%*+89<б9
2б/
)і&
input_1         dd
p 
ф "і         Ћ
.__inference_texture_cnn_5_layer_call_fn_219951c
$%*+89;б8
1б.
(і%
inputs         dd
p 
ф "і         Ћ
.__inference_texture_cnn_5_layer_call_fn_219976c
$%*+89;б8
1б.
(і%
inputs         dd
p
ф "і         ќ
.__inference_texture_cnn_5_layer_call_fn_220001d
$%*+89<б9
2б/
)і&
input_1         dd
p
ф "і         