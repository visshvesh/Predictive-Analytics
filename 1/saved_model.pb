Ë
ë
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	
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
5
Pow
x"T
y"T
z"T"
Ttype:
	2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.3.02
b'unknown'x
n
PlaceholderPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
p
Placeholder_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

$w/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      *
_class

loc:@w
~
#w/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    *
_class

loc:@w

%w/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?*
_class

loc:@w
Ě
.w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal$w/Initializer/truncated_normal/shape*
seed2 *

seed *
_output_shapes

:*
dtype0*
T0*
_class

loc:@w
ż
"w/Initializer/truncated_normal/mulMul.w/Initializer/truncated_normal/TruncatedNormal%w/Initializer/truncated_normal/stddev*
_output_shapes

:*
T0*
_class

loc:@w
­
w/Initializer/truncated_normalAdd"w/Initializer/truncated_normal/mul#w/Initializer/truncated_normal/mean*
_output_shapes

:*
T0*
_class

loc:@w

w
VariableV2*
shared_name *
	container *
shape
:*
_output_shapes

:*
dtype0*
_class

loc:@w

w/AssignAssignww/Initializer/truncated_normal*
validate_shape(*
_output_shapes

:*
T0*
_class

loc:@w*
use_locking(
T
w/readIdentityw*
_output_shapes

:*
T0*
_class

loc:@w
v
b/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class

loc:@b

b
VariableV2*
shared_name *
	container *
shape:*
_output_shapes
:*
dtype0*
_class

loc:@b

b/AssignAssignbb/Initializer/zeros*
validate_shape(*
_output_shapes
:*
T0*
_class

loc:@b*
use_locking(
P
b/readIdentityb*
_output_shapes
:*
T0*
_class

loc:@b
"
initNoOp	^w/Assign	^b/Assign
}
MatMulMatMulPlaceholderw/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0*
transpose_a( 
L
addAddMatMulb/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
P
subSubaddPlaceholder_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
H
powPowsubpow/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       
V
MeanMeanpowConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
r
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
\
gradients/Mean_grad/ShapeShapepow*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

Tmultiples0
^
gradients/Mean_grad/Shape_1Shapepow*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
gradients/pow_grad/ShapeShapesub*
_output_shapes
:*
T0*
out_type0
]
gradients/pow_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
´
(gradients/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pow_grad/Shapegradients/pow_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
s
gradients/pow_grad/mulMulgradients/Mean_grad/truedivpow/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
]
gradients/pow_grad/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
_
gradients/pow_grad/subSubpow/ygradients/pow_grad/sub/y*
_output_shapes
: *
T0
l
gradients/pow_grad/PowPowsubgradients/pow_grad/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pow_grad/mul_1Mulgradients/pow_grad/mulgradients/pow_grad/Pow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ą
gradients/pow_grad/SumSumgradients/pow_grad/mul_1(gradients/pow_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/pow_grad/ReshapeReshapegradients/pow_grad/Sumgradients/pow_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients/pow_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
z
gradients/pow_grad/GreaterGreatersubgradients/pow_grad/Greater/y*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
T
gradients/pow_grad/LogLogsub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
gradients/pow_grad/zeros_like	ZerosLikesub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¨
gradients/pow_grad/SelectSelectgradients/pow_grad/Greatergradients/pow_grad/Loggradients/pow_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
gradients/pow_grad/mul_2Mulgradients/Mean_grad/truedivpow*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/pow_grad/mul_3Mulgradients/pow_grad/mul_2gradients/pow_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
gradients/pow_grad/Sum_1Sumgradients/pow_grad/mul_3*gradients/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/pow_grad/Reshape_1Reshapegradients/pow_grad/Sum_1gradients/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
g
#gradients/pow_grad/tuple/group_depsNoOp^gradients/pow_grad/Reshape^gradients/pow_grad/Reshape_1
Ú
+gradients/pow_grad/tuple/control_dependencyIdentitygradients/pow_grad/Reshape$^gradients/pow_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/pow_grad/Reshape
Ď
-gradients/pow_grad/tuple/control_dependency_1Identitygradients/pow_grad/Reshape_1$^gradients/pow_grad/tuple/group_deps*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients/pow_grad/Reshape_1
[
gradients/sub_grad/ShapeShapeadd*
_output_shapes
:*
T0*
out_type0
g
gradients/sub_grad/Shape_1ShapePlaceholder_1*
_output_shapes
:*
T0*
out_type0
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
´
gradients/sub_grad/SumSum+gradients/pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/sub_grad/Sum_1Sum+gradients/pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ú
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
T0*
out_type0
d
gradients/add_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
´
gradients/add_grad/SumSum+gradients/sub_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
gradients/add_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ú
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
Ó
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
:*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
ł
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyw/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0*
transpose_a( 
ą
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_b( *
T0*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ä
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
b
GradientDescent/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *
×Ł;
ë
-GradientDescent/update_w/ApplyGradientDescentApplyGradientDescentwGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
T0*
_class

loc:@w*
use_locking( 
ä
-GradientDescent/update_b/ApplyGradientDescentApplyGradientDescentbGradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*
_class

loc:@b*
use_locking( 
w
GradientDescentNoOp.^GradientDescent/update_w/ApplyGradientDescent.^GradientDescent/update_b/ApplyGradientDescent

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
P

save/ConstConst*
_output_shapes
: *
dtype0*
valueB Bmodel

save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d56f2f66ffb14847bc003cb719436cf2/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
\
save/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
e
save/SaveV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBbBw
g
save/SaveV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueBB B 
{
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbw*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*

axis *
T0*
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
_output_shapes
: *
T0
e
save/RestoreV2/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBb
h
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes
:*
dtypes
2

save/AssignAssignbsave/RestoreV2*
validate_shape(*
_output_shapes
:*
T0*
_class

loc:@b*
use_locking(
g
save/RestoreV2_1/tensor_namesConst*
_output_shapes
:*
dtype0*
valueBBw
j
!save/RestoreV2_1/shape_and_slicesConst*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
_output_shapes
:*
dtypes
2

save/Assign_1Assignwsave/RestoreV2_1*
validate_shape(*
_output_shapes

:*
T0*
_class

loc:@w*
use_locking(
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"E
	variables86

w:0w/Assignw/read:0

b:0b/Assignb/read:0"
train_op

GradientDescent"$
legacy_init_op

legacy_init_op"O
trainable_variables86

w:0w/Assignw/read:0

b:0b/Assignb/read:0*

predictions
-
input$
Placeholder:0˙˙˙˙˙˙˙˙˙&
output
add:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict