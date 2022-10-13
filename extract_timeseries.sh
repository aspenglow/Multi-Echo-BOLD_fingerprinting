#!/usr/bin/env bash

displayhelp() {
echo "Usage:"
echo "./extract_timeseries.sh -atlas {atlas} -mref {mref} [-overwrite]"
echo ""
echo "Required arguments:"
echo "atlas mref"
echo "Optional arguments:"
echo "overwrite"
exit ${1:-0}
}

# Check if there is input

if [[ ( $# -eq 0 ) ]]
	then
	displayhelp
fi

### print input
printline=$( basename -- $0 )
echo "${printline} " "$@"

# Parsing required and optional variables with flags
# Also checking if a flag is the help request or the version
overwrite=no

while [ ! -z "$1" ]
do
	case "$1" in
		-atlas)		atlas=$2;shift;;
		-mref)		mref=$2;shift;;

		-overwrite) overwrite="yes";;
		-h)			displayhelp;;
		-v)			version;exit 0;;
		*)			echo "Wrong flag: $1";displayhelp 1;;
	esac
	shift
done

### Remove nifti suffix
for var in atlas mref
do
	echo "${var} is set to ${!var}"
	eval "${var}=${!var%.nii*}"
done
echo "overwrite is set to ${overwrite}"

######################################
######### Script starts here #########
######################################

cwd=$(pwd)

#Read and process input
rdir=$( dirname ${mref} )
mref=$( basename ${mref} )
subatlas=$( basename ${atlas} )2mref

mrefsfx=${mref#sub-*_}
mrefsub=${mref%$mrefsfx}

cd ${rdir} || exit

echo "Bring atlas $( basename ${atlas}) to subject space"

if [[ ! -e ${subatlas}.nii.gz ]] || [[ "${overwrite}" == "yes" ]]
then
	antsApplyTransforms -d 3 -i ${atlas}.nii.gz \
	-r ${mref}.nii.gz -o ${subatlas}.nii.gz \
	-n MultiLabel \
	-t ${mrefsub}T1w2mref0GenericAffine.mat \
	-t [${mrefsub}T1w2std0GenericAffine.mat,1] \
	-t ${mrefsub}T1w2std1InverseWarp.nii.gz
fi

echo "Check if existing or create folder"
mkdir -p ../../extracted_timeseries

cd ../func || exit

for f in 00.sub-*preprocessed.nii.gz
do
	fname=${f#00.}
	fname=${fname%_native*}_$( basename ${atlas%.nii*})
	echo "Extract timeseries from ${f} with $( basename ${atlas})"
	ts=../../extracted_timeseries/${fname}.txt
	fslmeants -i ${f} --label=${rdir}/${subatlas}.nii.gz --transpose > ${ts}
done

cd ${cwd}
