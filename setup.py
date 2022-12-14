from setuptools import find_packages, setup


def main():
    package_name = 'make_it_talk'
    packages = find_packages(package_name)
    packages = list(map(lambda x: f'{package_name}/{x}', packages))

    setup(
        name=package_name,
        version='0.0.1',
        author='voidmax',
        description=package_name,
        package_dir={package_name: package_name},
        packages=packages,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
        install_requires=[
            'torch>=1.11.0',
            'resemblyzer',
            'face_alignment',
            'pysptk', 
            'scikit-image',
            'soundfile',
            'librosa'
        ],
    )


if __name__ == '__main__':
    main()