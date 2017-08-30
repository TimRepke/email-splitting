package jangada;

import edu.cmu.minorthird.text.*;

/**
 * An abstract annotator that is based on marking up substrings within
 * a string, using the CharAnnotation class from Minorthird.
 *
 * @author ksteppe
 */

public abstract class LocalStringAnnotator extends AbstractAnnotator
{
	protected String providedAnnotation = null;

  public void doAnnotate(MonotonicTextLabels labels)
	{
		//add the annotations into labels
		TextBase textBase = labels.getTextBase();
		for (Span.Looper it = textBase.documentSpanIterator(); it.hasNext();)
		{
			Span span = it.nextSpan();
			String spanString = span.asString();

      CharAnnotation[] annotations = annotateString(spanString);

      if (annotations != null)
      {
        for (int i = 0; i < annotations.length; i++)
        {
          CharAnnotation ann = annotations[i];
          int lo = ann.getOffset();
          Span newSpan = span.charIndexSubSpan(lo, lo + ann.getLength());
          labels.addToType(newSpan, ann.getType());
        }
      }
		}
		if (providedAnnotation!=null) labels.setAnnotatedBy(providedAnnotation);
	}

  protected String[] closedTypes()
  {
    return null;
  }

	/** Override this class to provide the actual annotations for a span.
	 */
  protected abstract CharAnnotation[] annotateString(String spanString);

}
