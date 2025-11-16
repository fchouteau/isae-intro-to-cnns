serve-slides:
	cd docs; reveal-md . \
				-w \
				--disable-auto-open \
				--template static/template/reveal.html \
				--static-dirs static

build-slides:
	rm -rf ../docs || true
	cd docs; reveal-md . \
				--static ../../docs \
				--template static/template/reveal.html \
				--static-dirs static
	rm ../docs/index.html && mv ../docs/slides.html ../docs/index.html

build-student-notebooks:
	jupytext --to notebook 0_cnns.py --output ../0_cnns.ipynb
	jupytext --to notebook 1_hands_on.py --output ../1_hands_on.ipynb
	jupytext --to notebook 2_sliding_window.py --output ../2_sliding_window.ipynb
	jupytext --to notebook 3_more_complex_example.py --output ../3_more_complex_example.ipynb
#	jupytext --to notebook 4_classification_to_dense.py --output ../4_classification_to_dense.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../0_cnns.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../1_hands_on.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../2_sliding_window.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../3_more_complex_example.ipynb

#	jupyter nbconvert --to notebook \
#	    --TagRemovePreprocessor.enabled=True \
#	    --TagRemovePreprocessor.remove_cell_tags=solution \
#	    --inplace \
#        ../4_classification_to_dense.ipynb

copy-to-isae-repo:
	cp ../README.md /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../licence.md /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../0_cnns.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../1_hands_on.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../2_sliding_window.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../3_more_complex_example.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
#	cp ../4_classification_to_dense.ipynb /home/fchouteau/classes/deep-learning/vision/
