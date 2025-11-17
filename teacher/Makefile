serve-slides:
	cd docs; reveal-md . \
				-w \
				--disable-auto-open \
				--template static/template/reveal.html \
				--listing-template static/template/listing.html \
				--static-dirs static

build-slides:
	rm -rf ../docs || true
	cd docs; reveal-md . \
				--static ../../docs \
				--template static/template/reveal.html \
				--listing-template static/template/listing.html \
				--static-dirs static

build-student-notebooks:
	jupytext --to notebook 0_cnns.py --output ../0_cnns.ipynb
	jupytext --to notebook 1_hands_on.py --output ../1_hands_on.ipynb
	jupytext --to notebook 2_sliding_window.py --output ../2_sliding_window.ipynb
	jupytext --to notebook 3_classification_to_segmentation.py --output ../3_classification_to_segmentation.ipynb
	jupytext --to notebook 4_sliding_window_unet.py --output ../4_sliding_window_unet.ipynb

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
        ../3_classification_to_segmentation.ipynb

	jupyter nbconvert --to notebook \
	    --TagRemovePreprocessor.enabled=True \
	    --TagRemovePreprocessor.remove_cell_tags=solution \
	    --inplace \
        ../4_sliding_window_unet.ipynb

copy-to-isae-repo:
	cp ../README.md /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../licence.md /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../0_cnns.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../1_hands_on.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../2_sliding_window.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../3_classification_to_segmentation.ipynb /Users/fchouteau/repositories/classes/deep-learning/vision/
	cp ../4_sliding_window_unet.ipynb /home/fchouteau/classes/deep-learning/vision/
