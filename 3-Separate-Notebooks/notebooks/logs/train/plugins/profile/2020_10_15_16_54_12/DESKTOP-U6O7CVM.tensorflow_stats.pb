"�C
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1    `:AA    `:Aa���V��?i���V��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff��@9fffff��@Afffff��@Ifffff��@a��mӧ�?i�[���?�Unknown�
sHostDestroyResourceOp"DestroyResourceOp(1fffff�M@9�Ő�[ @Afffff�M@I�Ő�[ @a"�RI,?i���Y��?�Unknown�
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1�����YA@9�����YA@A�����YA@I�����YA@a5,�,� ?i�D��b��?�Unknown
iHostWriteSummary"WriteSummary(1     �<@9     �<@A     �<@I     �<@a�.!g30?i�}E@<��?�Unknown�
mHostSoftmax"model_2/dense_22/Softmax(1�����:@9�����:@A�����:@I�����:@a�Pw �?it��p��?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1333333:@9333333:@A      7@I      7@a	���?iS �����?�Unknown
d	HostDataset"Iterator::Model(1fffff�;@9fffff�;@A�����5@I�����5@aql��� ?i�� T��?�Unknown
[
HostAddV2"Adam/add(1������3@9������3@A������3@I������3@afi���?i�L����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(13333333@93333333@A      /@I      /@a#!���?i��ga��?�Unknown
oHost_FusedMatMul"model_2/dense_20/Relu(1������(@9������(@A������(@I������(@a�Mۄ��?i�H
���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1333333%@9333333%@A333333%@I333333%@a]�ܚi9?i�c����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1333333 @9333333 @A333333 @I333333 @a8�9���>i��;�N��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333@9333333@A333333@I333333@aw�1'��>i :�w���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@aw�1'��>i%��-���?�Unknown�
yHostMatMul"%gradient_tape/model_2/dense_22/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a�n��>i�yg]���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333@9333333@A333333@I333333@a�+�˷��>i��B(��?�Unknown
{HostMatMul"'gradient_tape/model_2/dense_22/MatMul_1(1333333@9333333@A333333@I333333@a�+�˷��>i��F(\��?�Unknown
�HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1ffffff@9ffffff@Affffff@Iffffff@a��L	X/�>i�������?�Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@a�j �!�>iZ��ʺ��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1333333@9333333@A333333@I333333@a]�ܚi9�>i�{=���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      @9      @A      @I      @a@���Y�>i"�/f	��?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1������@9������@A������@I������@a�Ak���>i���.��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a��N5�P�>i�wxmS��?�Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a·VJ�>i��ηr��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     �H@9     �H@Affffff@Iffffff@a·VJ�>i��$���?�Unknown
{HostSum"*categorical_crossentropy/weighted_loss/Sum(1ffffff@9ffffff@Affffff@Iffffff@a·VJ�>i��zL���?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a,
/�<�>i��Q����?�Unknown
rHost_FusedMatMul"model_2/dense_22/BiasAdd(1������@9������@A������@I������@a,
/�<�>i+�(����?�Unknown
�HostReadVariableOp"&model_2/dense_22/MatMul/ReadVariableOp(1������@9������@A������@I������@a,
/�<�>iZ����?�Unknown
y HostMatMul"%gradient_tape/model_2/dense_21/MatMul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a��L	X/�>i�X2��?�Unknown
Y!HostPow"Adam/Pow(1������	@9������	@A������	@I������	@a�G�k�>i�OP�7��?�Unknown
�"HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1������	@9������	@A������	@I������	@a�G�k�>iϖH
P��?�Unknown
y#HostMatMul"%gradient_tape/model_2/dense_20/MatMul(1������@9������@A������@I������@a�Mۄ���>i��g��?�Unknown
{$HostMatMul"'gradient_tape/model_2/dense_21/MatMul_1(1������@9������@A������@I������@a�Mۄ���>i��y[��?�Unknown
�%HostBiasAddGrad"2gradient_tape/model_2/dense_22/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�j �!�>i�R}���?�Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@a�0�����>i�ZU���?�Unknown
X'HostCast"Cast_1(1      @9      @A      @I      @a@���Y�>i7Rfi���?�Unknown
\(HostArgMax"ArgMax_1(1333333@9333333@A333333@I333333@a��N5�P�>i��`����?�Unknown
V)HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@anRs���>i���G���?�Unknown
o*Host_FusedMatMul"model_2/dense_21/Relu(1ffffff@9ffffff@Affffff@Iffffff@anRs���>i�m�����?�Unknown
v+HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1������@9������@A������@I������@a�ݰ:��>i�П��?�Unknown
[,HostPow"
Adam/Pow_1(1������ @9������ @A������ @I������ @a�����>i5����?�Unknown
v-HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @af��X���>i�9&�!��?�Unknown
�.HostDivNoNan",categorical_crossentropy/weighted_loss/value(1       @9       @A       @I       @af��X���>if�-1��?�Unknown
�/HostBiasAddGrad"2gradient_tape/model_2/dense_21/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��g�6 �>iAм�?��?�Unknown
]0HostCast"Adam/Cast_1(1�������?9�������?A�������?I�������?a�j�Owy�>i<xxjM��?�Unknown
}1HostReluGrad"'gradient_tape/model_2/dense_21/ReluGrad(1�������?9�������?A�������?I�������?a�j�Owy�>i7 4'[��?�Unknown
X2HostEqual"Equal(1�������?9�������?A�������?I�������?a�G�k�>i�C0]g��?�Unknown
`3HostGatherV2"
GatherV2_1(1�������?9�������?A�������?I�������?a�G�k�>iKg,�s��?�Unknown
t4HostAssignAddVariableOp"AssignAddVariableOp(1      �?9      �?A      �?I      �?aM���8��>i�����?�Unknown
T5HostMul"Mul(1�������?9�������?A�������?I�������?a�0�����>i|�����?�Unknown
�6HostBiasAddGrad"2gradient_tape/model_2/dense_20/BiasAdd/BiasAddGrad(1�������?9�������?A�������?I�������?a�0�����>i\��ݒ��?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a��N5�P�>i�����?�Unknown
b8HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a��N5�P�>i��|.���?�Unknown
}9HostReluGrad"'gradient_tape/model_2/dense_20/ReluGrad(1      �?9      �?A      �?I      �?af��X���>i�M:Ь��?�Unknown
o:HostReadVariableOp"Adam/ReadVariableOp(1�������?9�������?A�������?I�������?a�j�Owy�>i�!�����?�Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a�j�Owy�>i�������?�Unknown
~<HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�G�k�>i������?�Unknown
t=HostReadVariableOp"Adam/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�G�k�>if�����?�Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�G�k�>i++�����?�Unknown
a?HostIdentity"Identity(1�������?9�������?A�������?I�������?a�G�k�>i�<�����?�Unknown�
�@HostReadVariableOp"'model_2/dense_20/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�G�k�>i�N����?�Unknown
`AHostDivNoNan"
div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?azo1>y^�>iA��k���?�Unknown
�BHostReadVariableOp"&model_2/dense_20/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?azo1>y^�>i��(����?�Unknown
�CHostReadVariableOp"'model_2/dense_21/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?azo1>y^�>iY=����?�Unknown
�DHostReadVariableOp"'model_2/dense_22/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?azo1>y^�>i�er���?�Unknown
yEHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a��N5�P�>i9����?�Unknown
�FHostReadVariableOp"&model_2/dense_21/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��N5�P�>i������?�Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�G�k�>io�a����?�Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�G�k�>iQ�����?�Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��N5�P�>i�������?�Unknown*�B
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff��@9fffff��@Afffff��@Ifffff��@a�3�r��?i�3�r��?�Unknown�
sHostDestroyResourceOp"DestroyResourceOp(1fffff�M@9�Ő�[ @Afffff�M@I�Ő�[ @ar[V�A{?i�Vs㡷�?�Unknown�
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1�����YA@9�����YA@A�����YA@I�����YA@a�U+�o?i�m����?�Unknown
iHostWriteSummary"WriteSummary(1     �<@9     �<@A     �<@I     �<@a�szW3j?if��#���?�Unknown�
mHostSoftmax"model_2/dense_22/Softmax(1�����:@9�����:@A�����:@I�����:@a����F�g?i���j�	�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1333333:@9333333:@A      7@I      7@a[��$e?iKz&��?�Unknown
dHostDataset"Iterator::Model(1fffff�;@9fffff�;@A�����5@I�����5@a0�lU�ec?i��ϾC2�?�Unknown
[HostAddV2"Adam/add(1������3@9������3@A������3@I������3@a�Gm�3b?iL�<gwD�?�Unknown
�	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(13333333@93333333@A      /@I      /@a�k�l\?i���R�?�Unknown
o
Host_FusedMatMul"model_2/dense_20/Relu(1������(@9������(@A������(@I������(@a\�s�V�V?iGA�H^�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1333333%@9333333%@A333333%@I333333%@aZ� � }S?i��f��g�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1333333 @9333333 @A333333 @I333333 @a����M?i̪No�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1333333@9333333@A333333@I333333@a� g���J?i�%v�?�Unknown
eHost
LogicalAnd"
LogicalAnd(1333333@9333333@A333333@I333333@a� g���J?i�w��|�?�Unknown�
yHostMatMul"%gradient_tape/model_2/dense_22/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a\_ƻ�J?iq��@��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333@9333333@A333333@I333333@a\mUo%I?ivF(���?�Unknown
{HostMatMul"'gradient_tape/model_2/dense_22/MatMul_1(1333333@9333333@A333333@I333333@a\mUo%I?i�^q���?�Unknown
�HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1ffffff@9ffffff@Affffff@Iffffff@a̴��DH?i���ҕ�?�Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@a�F2�SE?i�Uɜ'��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1333333@9333333@A333333@I333333@aZ� � }C?iÝ���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      @9      @A      @I      @aZ����bB?i��n����?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1������@9������@A������@I������@a�p�5�B?i�A<� ��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a �t�A?iWnV���?�Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a�ֹ~'>?i��]7O��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     �H@9     �H@Affffff@Iffffff@a�ֹ~'>?iͳM��?�Unknown
{HostSum"*categorical_crossentropy/weighted_loss/Sum(1ffffff@9ffffff@Affffff@Iffffff@a�ֹ~'>?i�=�ظ�?�Unknown
�HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a]Q7�5;?i򑾷?��?�Unknown
rHost_FusedMatMul"model_2/dense_22/BiasAdd(1������@9������@A������@I������@a]Q7�5;?iܘ?v���?�Unknown
�HostReadVariableOp"&model_2/dense_22/MatMul/ReadVariableOp(1������@9������@A������@I������@a]Q7�5;?iƟ�4��?�Unknown
yHostMatMul"%gradient_tape/model_2/dense_21/MatMul(1ffffff
@9ffffff
@Affffff
@Iffffff
@a̴��D8?i`�����?�Unknown
YHostPow"Adam/Pow(1������	@9������	@A������	@I������	@a�*���7?i�XI���?�Unknown
� HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1������	@9������	@A������	@I������	@a�*���7?ijۿ����?�Unknown
y!HostMatMul"%gradient_tape/model_2/dense_20/MatMul(1������@9������@A������@I������@a\�s�V�6?i۩�����?�Unknown
{"HostMatMul"'gradient_tape/model_2/dense_21/MatMul_1(1������@9������@A������@I������@a\�s�V�6?iLxu���?�Unknown
�#HostBiasAddGrad"2gradient_tape/model_2/dense_22/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�F2�S5?i���U��?�Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1������@9������@A������@I������@a�bP��3?i�(�f���?�Unknown
X%HostCast"Cast_1(1      @9      @A      @I      @aZ����b2?i�����?�Unknown
\&HostArgMax"ArgMax_1(1333333@9333333@A333333@I333333@a �t�1?i}���:��?�Unknown
V'HostSum"Sum_2(1ffffff@9ffffff@Affffff@Iffffff@a�~n�/�0?iMγ�W��?�Unknown
o(Host_FusedMatMul"model_2/dense_21/Relu(1ffffff@9ffffff@Affffff@Iffffff@a�~n�/�0?i��u��?�Unknown
v)HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1������@9������@A������@I������@aZ���-0?i�u�z��?�Unknown
[*HostPow"
Adam/Pow_1(1������ @9������ @A������ @I������ @a	xZ\L�.?i�;�
i��?�Unknown
v+HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1       @9       @A       @I       @a]5��j-?iM�?��?�Unknown
�,HostDivNoNan",categorical_crossentropy/weighted_loss/value(1       @9       @A       @I       @a]5��j-?i�^,c��?�Unknown
�-HostBiasAddGrad"2gradient_tape/model_2/dense_21/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a����8�+?i&������?�Unknown
].HostCast"Adam/Cast_1(1�������?9�������?A�������?I�������?a��*�y*?i�e�!}��?�Unknown
}/HostReluGrad"'gradient_tape/model_2/dense_21/ReluGrad(1�������?9�������?A�������?I�������?a��*�y*?i���$��?�Unknown
X0HostEqual"Equal(1�������?9�������?A�������?I�������?a�*���'?i?P[F���?�Unknown
`1HostGatherV2"
GatherV2_1(1�������?9�������?A�������?I�������?a�*���'?i�����?�Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1      �?9      �?A      �?I      �?a���&?i�6�v��?�Unknown
T3HostMul"Mul(1�������?9�������?A�������?I�������?a�bP��#?i�C����?�Unknown
�4HostBiasAddGrad"2gradient_tape/model_2/dense_20/BiasAdd/BiasAddGrad(1�������?9�������?A�������?I�������?a�bP��#?i�h����?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1333333�?9333333�?A333333�?I333333�?a �t�!?i��R���?�Unknown
b6HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a �t�!?i�J���?�Unknown
}7HostReluGrad"'gradient_tape/model_2/dense_20/ReluGrad(1      �?9      �?A      �?I      �?a]5��j?ikS�����?�Unknown
o8HostReadVariableOp"Adam/ReadVariableOp(1�������?9�������?A�������?I�������?a��*�y?i!�-����?�Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a��*�y?i���p���?�Unknown
~:HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�*���?ix���^��?�Unknown
t;HostReadVariableOp"Adam/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�*���?i>b���?�Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a�*���?i��??���?�Unknown
a=HostIdentity"Identity(1�������?9�������?A�������?I�������?a�*���?i[����?�Unknown�
�>HostReadVariableOp"'model_2/dense_20/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�*���?i���O��?�Unknown
`?HostDivNoNan"
div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?aZ��=��?i�=����?�Unknown
�@HostReadVariableOp"&model_2/dense_20/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aZ��=��?i�~A���?�Unknown
�AHostReadVariableOp"'model_2/dense_21/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aZ��=��?i����=��?�Unknown
�BHostReadVariableOp"'model_2/dense_22/BiasAdd/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aZ��=��?i0�����?�Unknown
yCHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a �t�?i�
��o��?�Unknown
�DHostReadVariableOp"&model_2/dense_21/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a �t�?i"CO!���?�Unknown
wEHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�*���?is�C[��?�Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�*���?i��,f���?�Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a �t�?i      �?�Unknown